// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/bandit.hpp"
#include <cmath>
#include <map>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of the bandit algorithms (Sutton & Barto, Chapter 2):
 * - `ucb`: the bounds and choices worked out by hand, untried actions first, ties at random
 * - `gradient_bandit`: the preferences and probabilities worked out by hand, with and
 *   without the baseline, a softmax that does not overflow, and actions drawn from π
 * - the 10-armed testbed: UCB against ε-greedy (Figure 2.4), and the gradient bandit
 *   with and without the baseline (Figure 2.5)
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;

using game_type = games::bandit;
using types     = games::types<game_type>;

/// @return one pull of arm @param a of the bandit "b", which paid @param reward
episode pull(const std::string & a,
             double reward)
{
    return episode {{state(reward, "b"), action(a)}};
}

/// @return the arms of @param game, as relearn actions
std::vector<types::action> arms_of(const game_type & game)
{
    std::vector<types::action> arms;
    for (auto a : game.actions(0)) {
        arms.push_back(types::action(a));
    }
    return arms;
}

/**
 * @brief UCB on the 10-armed testbed (Figure 2.4)
 * @param c is the degree of exploration
 * @param runs is the number of bandits, each one learnt from scratch
 * @param pulls is the number of steps of each run
 * @param first_seed seeds the first bandit; the next ones use the seeds after it
 * @return the average reward at each step 1 ... pulls (index 0 is unused)
 */
std::vector<double> ucb_rewards(double c,
                                unsigned int runs,
                                unsigned int pulls,
                                unsigned int first_seed)
{
    std::vector<double> reward(pulls + 1, 0);
    for (unsigned int run = 0; run < runs; run++) {
        game_type game(10, first_seed + run);
        auto arms = arms_of(game);
        types::policy memory;
        relearn::ucb<types::state, types::action> learner(c, first_seed + run + 1000);
        auto agent = [&](const unsigned int & s,
                         const std::vector<unsigned int> &) {
            return learner.choose(types::state(s), arms, memory).trait();
        };
        for (unsigned int i = 1; i <= pulls; i++) {
            auto r = games::play(game, agent, 1);
            reward[i] += r.total_reward / runs;
            learner(r.episode, memory);
        }
    }
    return reward;
}

/**
 * @brief ε-greedy with sample averages on the 10-armed testbed (Figure 2.4), on the same bandits as `ucb_rewards`
 * @param epsilon is the probability of exploring
 * @param runs is the number of bandits
 * @param pulls is the number of steps of each run
 * @param first_seed seeds the first bandit
 * @return the average reward at each step 1 ... pulls (index 0 is unused)
 */
std::vector<double> epsilon_greedy_rewards(double epsilon,
                                           unsigned int runs,
                                           unsigned int pulls,
                                           unsigned int first_seed)
{
    std::vector<double> reward(pulls + 1, 0);
    for (unsigned int run = 0; run < runs; run++) {
        game_type game(10, first_seed + run);
        std::mt19937 prng(first_seed + run + 1000);
        types::policy memory;
        relearn::q_probabilistic<types::state, types::action> learner(1.0);
        auto agent = [&](const unsigned int & s,
                         const std::vector<unsigned int> & arms) {
            return games::epsilon_greedy<game_type>(memory, s, arms, epsilon, prng);
        };
        for (unsigned int i = 1; i <= pulls; i++) {
            auto r = games::play(game, agent, 1);
            reward[i] += r.total_reward / runs;
            learner(r.episode, memory, r.terminated);
        }
    }
    return reward;
}

/**
 * @brief the gradient bandit on the 10-armed testbed with means around +4 (Figure 2.5)
 * @param alpha is the step size
 * @param baseline is true to use the average reward as the baseline
 * @param runs is the number of bandits
 * @param pulls is the number of steps of each run
 * @param first_seed seeds the first bandit
 * @return the share of runs pulling the best arm at each step 1 ... pulls (index 0 is unused)
 */
std::vector<double> gradient_optimal(double alpha,
                                     bool baseline,
                                     unsigned int runs,
                                     unsigned int pulls,
                                     unsigned int first_seed)
{
    std::vector<double> optimal(pulls + 1, 0);
    for (unsigned int run = 0; run < runs; run++) {
        game_type game(10, first_seed + run, 4.0);
        auto arms = arms_of(game);
        types::policy preferences;
        relearn::gradient_bandit<types::state, types::action> learner(alpha, baseline, first_seed + run + 1000);
        auto agent = [&](const unsigned int & s,
                         const std::vector<unsigned int> &) {
            return learner.choose(types::state(s), arms, preferences).trait();
        };
        for (unsigned int i = 1; i <= pulls; i++) {
            auto r = games::play(game, agent, 1);
            if (r.episode[0].action.trait() == game.best_arm()) {
                optimal[i] += 1.0 / runs;
            }
            learner(r.episode, preferences, arms);
        }
    }
    return optimal;
}

/// @return the average of @param values[from] ... @param values[to]
double mean(const std::vector<double> & values,
            unsigned int from,
            unsigned int to)
{
    double sum = 0;
    for (unsigned int i = from; i <= to; i++) {
        sum += values[i];
    }
    return sum / (to - from + 1);
}
} // namespace

/// verifies the sample averages and the counts N(a) that UCB keeps
TEST_CASE("ucb: values are sample averages", "[bandits][ucb]")
{
    relearn::ucb<state,action> learner(1.0);
    policy memory;
    // arm x pays 1, then 0, then 2: its value is 1, then 1/2, then 1
    learner(pull("x", 1), memory);
    REQUIRE(memory.value(state("b"), action("x")) == Approx(1));
    learner(pull("x", 0), memory);
    REQUIRE(memory.value(state("b"), action("x")) == Approx(0.5));
    learner(pull("x", 2), memory);
    REQUIRE(memory.value(state("b"), action("x")) == Approx(1));
    REQUIRE(learner.visits.count(state("b"), action("x")) == 3);
    // an episode of several links is several pulls
    learner(episode {{state(4, "b"), action("y")},
                     {state(2, "b"), action("y")}}, memory);
    REQUIRE(memory.value(state("b"), action("y")) == Approx(3));
    REQUIRE(learner.visits.count(state("b"), action("y")) == 2);
}

/// verifies UCB's bounds and choices on a tiny bandit, worked out by hand
TEST_CASE("ucb: the choices worked out by hand", "[bandits][ucb]")
{
    // three arms which always pay the same: a pays 1, b pays 0, c pays 0.5; and c = 1
    std::map<std::string,double> pays = {{"a", 1.0}, {"b", 0.0}, {"c", 0.5}};
    std::vector<action> arms = {action("a"), action("b"), action("c")};
    relearn::ucb<state,action> learner(1.0, 7);
    policy memory;
    // steps 1 to 3: every arm untried counts as maximising, so all three are pulled first, in a random order
    std::map<std::string,int> first;
    for (int t = 1; t <= 3; t++) {
        auto a = learner.choose(state("b"), arms, memory);
        first[a.trait()]++;
        learner(pull(a.trait(), pays[a.trait()]), memory);
    }
    REQUIRE(first.size() == 3);
    // step 4: N = 1 for every arm, and Q is what each one paid, so the bounds are Q + sqrt(ln 4 / 1):
    //    a: 1 + 1.17741 = 2.17741    b: 0 + 1.17741 = 1.17741    c: 0.5 + 1.17741 = 1.67741
    REQUIRE(learner.bound(state("b"), action("a"), 4, memory) == Approx(1 + std::sqrt(std::log(4.0))));
    REQUIRE(learner.bound(state("b"), action("b"), 4, memory) == Approx(std::sqrt(std::log(4.0))));
    REQUIRE(learner.bound(state("b"), action("c"), 4, memory) == Approx(0.5 + std::sqrt(std::log(4.0))));
    // then, step by step (the bounds of a, b, c; the highest is pulled):
    //    t =  4: 2.1774 1.1774 1.6774 -> a     (N = 2, 1, 1)
    //    t =  5: 1.8971 1.2686 1.7686 -> a     a's bonus is now sqrt(ln 5 / 2)
    //    t =  6: 1.7728 1.3386 1.8386 -> c     a has been pulled so often that c's bonus wins
    //    t =  7: 1.8054 1.3950 1.4864 -> a
    //    t =  8: 1.7210 1.4420 1.5197 -> a
    //    t =  9: 1.6629 1.4823 1.5481 -> a
    //    t = 10: 1.6195 1.5174 1.5730 -> a
    //    t = 11: 1.5853 1.5485 1.5950 -> c
    //    t = 12: 1.5958 1.5764 1.4101 -> a
    //    t = 13: 1.5662 1.6015 1.4247 -> b     even the worst arm is tried again, as ln t grows
    std::vector<std::string> expected = {"a", "a", "c", "a", "a", "a", "a", "c", "a", "b"};
    for (const auto & e : expected) {
        auto a = learner.choose(state("b"), arms, memory);
        REQUIRE(a.trait() == e);
        learner(pull(a.trait(), pays[a.trait()]), memory);
    }
    // after 13 pulls: a 8 times, b twice, c 3 times
    REQUIRE(learner.visits.count(state("b"), action("a")) == 8);
    REQUIRE(learner.visits.count(state("b"), action("b")) == 2);
    REQUIRE(learner.visits.count(state("b"), action("c")) == 3);
}

/// verifies that untried actions come first, in a random order, and that the order is seeded
TEST_CASE("ucb: untried actions first, ties broken at random", "[bandits][ucb]")
{
    std::vector<action> arms = {action("0"), action("1"), action("2"), action("3"), action("4")};
    // an action never taken has an infinite bound
    relearn::ucb<state,action> fresh;
    policy empty;
    REQUIRE(std::isinf(fresh.bound(state("b"), action("0"), 1, empty)));
    // the first five choices are the five arms, whatever they pay (even with c = 0)
    std::map<std::string,int> firsts;
    for (unsigned int seed = 0; seed < 20; seed++) {
        relearn::ucb<state,action> learner(0.0, seed);
        policy memory;
        std::map<std::string,int> tried;
        for (int t = 0; t < 5; t++) {
            auto a = learner.choose(state("b"), arms, memory);
            tried[a.trait()]++;
            if (t == 0) {
                firsts[a.trait()]++;
            }
            learner(pull(a.trait(), -10.0 * t), memory);
        }
        REQUIRE(tried.size() == 5);
    }
    // the first choice is a tie between all five arms, broken at random: over 20 seeds, more than one arm comes first
    REQUIRE(firsts.size() > 1);
    // the same seed makes the same choices
    relearn::ucb<state,action> one(2.0, 3);
    relearn::ucb<state,action> two(2.0, 3);
    policy memory_one;
    policy memory_two;
    for (int t = 0; t < 20; t++) {
        auto a = one.choose(state("b"), arms, memory_one);
        auto b = two.choose(state("b"), arms, memory_two);
        REQUIRE(a == b);
        one(pull(a.trait(), t % 3), memory_one);
        two(pull(b.trait(), t % 3), memory_two);
    }
}

/// verifies the preferences H and the probabilities π of the gradient bandit, worked out by hand
TEST_CASE("gradient_bandit: preferences with the baseline, worked out by hand", "[bandits][gradient]")
{
    // three arms a, b, c, α = 0.1, with the baseline R̄_t: the average of the rewards before step t, R̄_1 = R_1
    std::vector<action> arms = {action("a"), action("b"), action("c")};
    relearn::gradient_bandit<state,action> learner(0.1, true);
    policy H;
    // every preference starts at 0, so π = 1/3 each
    auto pi = learner.probabilities(state("b"), arms, H);
    REQUIRE(pi[0] == Approx(1.0 / 3));
    REQUIRE(pi[1] == Approx(1.0 / 3));
    REQUIRE(pi[2] == Approx(1.0 / 3));
    // step 1: a pays 1. R̄_1 = R_1 = 1, so R_1 - R̄_1 = 0 and nothing moves
    learner(pull("a", 1), H, arms);
    REQUIRE(H.value(state("b"), action("a")) == Approx(0).margin(1e-12));
    REQUIRE(H.value(state("b"), action("b")) == Approx(0).margin(1e-12));
    REQUIRE(H.value(state("b"), action("c")) == Approx(0).margin(1e-12));
    // step 2: b pays 0. R̄_2 = 1 (the average of R_1 alone), so R_2 - R̄_2 = -1, and π = 1/3 each:
    //    H(b) = 0 + 0.1 * (-1) * (1 - 1/3) = -1/15 = -0.0666667
    //    H(a) = H(c) = 0 - 0.1 * (-1) * 1/3 = 1/30 = 0.0333333
    learner(pull("b", 0), H, arms);
    REQUIRE(H.value(state("b"), action("a")) == Approx(1.0 / 30));
    REQUIRE(H.value(state("b"), action("b")) == Approx(-1.0 / 15));
    REQUIRE(H.value(state("b"), action("c")) == Approx(1.0 / 30));
    //    π(a) = π(c) = e^(1/30) / (2 e^(1/30) + e^(-1/15)) = 0.3442533, π(b) = 0.3114933
    pi = learner.probabilities(state("b"), arms, H);
    REQUIRE(pi[0] == Approx(0.3442533457));
    REQUIRE(pi[1] == Approx(0.3114933085));
    REQUIRE(pi[2] == Approx(0.3442533457));
    // step 3: a pays 2. R̄_3 = (1 + 0) / 2 = 0.5, so R_3 - R̄_3 = 1.5:
    //    H(a) =  1/30 + 0.1 * 1.5 * (1 - 0.3442533) =  0.1316953
    //    H(b) = -1/15 - 0.1 * 1.5 * 0.3114933       = -0.1133907
    //    H(c) =  1/30 - 0.1 * 1.5 * 0.3442533       = -0.0183047
    learner(pull("a", 2), H, arms);
    REQUIRE(H.value(state("b"), action("a")) == Approx(0.1316953315));
    REQUIRE(H.value(state("b"), action("b")) == Approx(-0.1133906629));
    REQUIRE(H.value(state("b"), action("c")) == Approx(-0.0183046685));
    pi = learner.probabilities(state("b"), arms, H);
    REQUIRE(pi[0] == Approx(0.3783085147));
    REQUIRE(pi[1] == Approx(0.2960783291));
    REQUIRE(pi[2] == Approx(0.3256131562));
    // the preferences always add up to 0, since the update of A_t and those of the others cancel out
    double sum = H.value(state("b"), action("a")) + H.value(state("b"), action("b")) + H.value(state("b"), action("c"));
    REQUIRE(sum == Approx(0).margin(1e-12));
    // and the average reward, the next baseline, is (1 + 0 + 2) / 3 = 1
    REQUIRE(learner.rewards[state("b")].mean == Approx(1));
    REQUIRE(learner.rewards[state("b")].count == 3);
}

/// verifies the gradient bandit without the baseline (R̄_t = 0), worked out by hand
TEST_CASE("gradient_bandit: preferences without the baseline, worked out by hand", "[bandits][gradient]")
{
    std::vector<action> arms = {action("a"), action("b"), action("c")};
    relearn::gradient_bandit<state,action> learner(0.1, false);
    policy H;
    // step 1: a pays 1, R - 0 = 1, π = 1/3 each:
    //    H(a) = 0.1 * (1 - 1/3) = 1/15,  H(b) = H(c) = -0.1 * 1/3 = -1/30
    learner(pull("a", 1), H, arms);
    REQUIRE(H.value(state("b"), action("a")) == Approx(1.0 / 15));
    REQUIRE(H.value(state("b"), action("b")) == Approx(-1.0 / 30));
    REQUIRE(H.value(state("b"), action("c")) == Approx(-1.0 / 30));
    // step 2: b pays 0, R - 0 = 0: nothing moves (with the baseline, a reward of 0 would have counted as bad)
    learner(pull("b", 0), H, arms);
    REQUIRE(H.value(state("b"), action("a")) == Approx(1.0 / 15));
    REQUIRE(H.value(state("b"), action("b")) == Approx(-1.0 / 30));
    // step 3: a pays 2, with π(a) = e^(1/15) / (e^(1/15) + 2 e^(-1/30)) = 0.3559131, π(b) = π(c) = 0.3220435:
    //    H(a) = 1/15 + 0.1 * 2 * (1 - 0.3559131) = 0.1954841
    //    H(b) = H(c) = -1/30 - 0.1 * 2 * 0.3220435 = -0.0977420
    auto pi = learner.probabilities(state("b"), arms, H);
    REQUIRE(pi[0] == Approx(0.3559130712));
    REQUIRE(pi[1] == Approx(0.3220434644));
    learner(pull("a", 2), H, arms);
    REQUIRE(H.value(state("b"), action("a")) == Approx(0.1954840524));
    REQUIRE(H.value(state("b"), action("b")) == Approx(-0.0977420262));
    REQUIRE(H.value(state("b"), action("c")) == Approx(-0.0977420262));
    pi = learner.probabilities(state("b"), arms, H);
    REQUIRE(pi[0] == Approx(0.4013312961));
    REQUIRE(pi[1] == Approx(0.2993343519));
    REQUIRE(pi[2] == Approx(0.2993343519));
}

/// verifies that the softmax does not overflow with large preferences
TEST_CASE("gradient_bandit: a stable softmax", "[bandits][gradient]")
{
    // e^1000 overflows a double, but the softmax only depends on the differences:
    // H = 1000, 1001, 999 gives the same π as 0, 1, -1, that is 1, e, 1/e over (1 + e + 1/e)
    std::vector<action> arms = {action("a"), action("b"), action("c")};
    relearn::gradient_bandit<state,action> learner;
    policy H;
    H.update(state("b"), action("a"), 1000);
    H.update(state("b"), action("b"), 1001);
    H.update(state("b"), action("c"), 999);
    auto pi = learner.probabilities(state("b"), arms, H);
    const double e = std::exp(1.0);
    const double total = 1 + e + 1 / e;
    REQUIRE(pi[0] == Approx(1 / total));
    REQUIRE(pi[1] == Approx(e / total));
    REQUIRE(pi[2] == Approx(1 / e / total));
    // and very negative preferences do not all underflow to 0
    H.update(state("b"), action("a"), -2000);
    H.update(state("b"), action("b"), -2000);
    H.update(state("b"), action("c"), -2000);
    pi = learner.probabilities(state("b"), arms, H);
    REQUIRE(pi[0] == Approx(1.0 / 3));
    REQUIRE(pi[2] == Approx(1.0 / 3));
}

/// verifies that the gradient bandit draws its actions with the probabilities π
TEST_CASE("gradient_bandit: actions are drawn from π", "[bandits][gradient]")
{
    // H = 0, ln 2, ln 4 gives π = 1/7, 2/7, 4/7
    std::vector<action> arms = {action("a"), action("b"), action("c")};
    relearn::gradient_bandit<state,action> learner(0.1, true, 5);
    policy H;
    H.update(state("b"), action("b"), std::log(2.0));
    H.update(state("b"), action("c"), std::log(4.0));
    std::map<std::string,double> drawn;
    const int draws = 70000;
    for (int i = 0; i < draws; i++) {
        drawn[learner.choose(state("b"), arms, H).trait()] += 1.0 / draws;
    }
    // a standard deviation of at most 0.002 for 70000 draws: 0.01 is 5 of them
    REQUIRE(drawn["a"] == Approx(1.0 / 7).margin(0.01));
    REQUIRE(drawn["b"] == Approx(2.0 / 7).margin(0.01));
    REQUIRE(drawn["c"] == Approx(4.0 / 7).margin(0.01));
}

/// verifies the constructors, and brace initialisation
TEST_CASE("bandits: constructors and brace initialisation", "[bandits]")
{
    relearn::ucb<state,action> ucb_defaults;
    REQUIRE(ucb_defaults.c == Approx(2));
    auto ucb_one = relearn::ucb<state,action>{1.0, 3};
    REQUIRE(ucb_one.c == Approx(1));
    relearn::gradient_bandit<state,action> gradient_defaults;
    REQUIRE(gradient_defaults.alpha == Approx(0.1));
    REQUIRE(gradient_defaults.baseline);
    auto no_baseline = relearn::gradient_bandit<state,action>{0.4, false, 3};
    REQUIRE(no_baseline.alpha == Approx(0.4));
    REQUIRE_FALSE(no_baseline.baseline);
}

/// verifies the +4 option of the bandit game, used by Figure 2.5
TEST_CASE("bandits: the testbed with means around +4", "[bandits][games]")
{
    // the same seed draws the same means, shifted by exactly 4, and the same best arm
    game_type plain(10, 9);
    game_type shifted(10, 9, 4.0);
    for (unsigned int a = 0; a < 10; a++) {
        REQUIRE(shifted.mean(a) == Approx(plain.mean(a) + 4));
    }
    REQUIRE(shifted.best_arm() == plain.best_arm());
}

/// verifies Figure 2.4: after the first steps, UCB (c = 2) earns more than ε-greedy (ε = 0.1)
TEST_CASE("ucb: earns more than ε-greedy on the 10-armed testbed (Figure 2.4)", "[bandits][ucb][testbed]")
{
    // 200 bandits, 1000 steps, the average reward of steps 101 to 1000.
    // Over 10 seeds, with libstdc++ and with libc++, UCB earned 0.086 to 0.113 more than ε-greedy
    const auto ucb = ucb_rewards(2.0, 200, 1000, 0);
    const auto greedy = epsilon_greedy_rewards(0.1, 200, 1000, 0);
    REQUIRE(mean(ucb, 101, 1000) - mean(greedy, 101, 1000) > 0.04);
}

/// verifies the spike of Figure 2.4 at step 11, when every arm has been tried once
TEST_CASE("ucb: the spike at step 11 (Figure 2.4)", "[bandits][ucb][testbed]")
{
    // steps 1 to 10 pull each untried arm once, at random, so they earn about 0 on average.
    // At step 11 every bonus is the same, sqrt(ln 11 / 1), so UCB pulls the arm that paid the most,
    // which is likely a good one; at step 12 that arm's bonus has shrunk, and another arm is pulled.
    // 2000 bandits. Over 10 seeds, with libstdc++ and with libc++: step 10 earned -0.08 to 0.06,
    // step 11 earned 1.03 to 1.14, and step 12 earned 0.11 to 0.22 less than step 11
    const auto reward = ucb_rewards(2.0, 2000, 12, 0);
    REQUIRE(std::abs(reward[10]) < 0.15);
    REQUIRE(reward[11] > 0.9);
    REQUIRE(reward[11] - reward[12] > 0.05);
}

/// verifies Figure 2.5: on means around +4, the gradient bandit needs its baseline
TEST_CASE("gradient_bandit: the baseline matters (Figure 2.5)", "[bandits][gradient][testbed]")
{
    // 200 bandits, 1000 steps, the share of runs pulling the best arm in steps 901 to 1000.
    // Over 10 seeds, with libstdc++ and with libc++: with the baseline, 0.80 to 0.87 (α = 0.1) and
    // 0.65 to 0.76 (α = 0.4); without it, 0.45 to 0.56 (α = 0.1) and 0.23 to 0.33 (α = 0.4).
    // The baseline was worth 0.31 to 0.39 (α = 0.1) and 0.36 to 0.50 (α = 0.4)
    const double with_01    = mean(gradient_optimal(0.1, true, 200, 1000, 0), 901, 1000);
    const double with_04    = mean(gradient_optimal(0.4, true, 200, 1000, 0), 901, 1000);
    const double without_01 = mean(gradient_optimal(0.1, false, 200, 1000, 0), 901, 1000);
    const double without_04 = mean(gradient_optimal(0.4, false, 200, 1000, 0), 901, 1000);
    REQUIRE(with_01 > 0.7);
    REQUIRE(with_04 > 0.55);
    REQUIRE(without_01 < 0.65);
    REQUIRE(without_04 < 0.45);
    REQUIRE(with_01 - without_01 > 0.2);
    REQUIRE(with_04 - without_04 > 0.25);
}
