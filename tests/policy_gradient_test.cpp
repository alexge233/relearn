// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/short_corridor.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of the policy gradient methods (Sutton & Barto, Chapter 13):
 * - `softmax_policy`: π(a | s) from the preferences, ∇ ln π, and sampling
 * - `reinforce`, `reinforce_with_baseline` and `actor_critic`: updates worked out by hand,
 *   including the γ^t factor
 * - the short corridor (Example 13.1): REINFORCE (Figure 13.1), REINFORCE with a baseline
 *   (Figure 13.2), and Actor-Critic, against the exact value of the policy learnt
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;

/// @return the legal actions of the hand-worked tests: left and right, in every state
std::vector<action> left_right(const state &)
{
    return {action("left"), action("right")};
}

using corridor       = games::short_corridor;
using corridor_types = games::types<corridor>;
using corridor_reinforce = relearn::reinforce<corridor_types::state, corridor_types::action>;
using corridor_baseline  = relearn::reinforce_with_baseline<corridor_types::state, corridor_types::action>;
using corridor_critic    = relearn::actor_critic<corridor_types::state, corridor_types::action>;

/// @return the legal actions of the short corridor, for the learners: left and right, everywhere
std::vector<corridor_types::action> corridor_actions(const corridor_types::state &)
{
    return {corridor_types::action(corridor::go_left), corridor_types::action(corridor::go_right)};
}

/// what was learnt in the short corridor, over several runs
struct corridor_runs
{
    /// π(right) learnt in each run (in the first cell, when the cells are shown)
    std::vector<double> right;
    /// the exact value of the start, v(S), of the policy learnt in each run
    std::vector<double> values;
};

/**
 * @brief REINFORCE, with or without a baseline: play a whole episode, then learn from it
 * @return the played episode
 */
template <class learner_type,
          class chooser>
games::played<corridor> learn_episode(corridor & game,
                                      chooser && agent,
                                      learner_type & learner,
                                      corridor_types::policy & preferences)
{
    auto r = games::play(game, agent, 1000);
    learner(r.episode, preferences, r.terminated);
    return r;
}

/**
 * @brief Actor-Critic: play an episode, learning after every step
 * @return the played episode
 */
template <class chooser>
games::played<corridor> learn_episode(corridor & game,
                                      chooser && agent,
                                      corridor_critic & learner,
                                      corridor_types::policy & preferences)
{
    return games::play_online_actor_critic(game, agent, learner, preferences, 1000);
}

/**
 * @brief learn the short corridor, from scratch, in several runs
 * @param fresh is the learner each run starts from (its step sizes, γ and legal actions)
 * @param shown is true for the agent to see which cell it is in
 * @param episodes is the number of episodes of each run (each one cut short after 1000 steps)
 * @param runs is the number of runs
 * @param seed seeds the runs: run i is seeded with 1000 * seed + i
 * @return the policy learnt in each run, and its exact value
 *
 * When the cells look the same, every run starts from π(right) = 0.05, the ε-greedy-left
 * policy (ε = 0.1) of Example 13.1, worth about -82: h(right) = -ln(19) / 2, h(left) = ln(19) / 2.
 * When they are shown, every run starts from the preferences 0, π(right) = 1/2 in every cell.
 */
template <class learner_type>
corridor_runs learn_corridor(const learner_type & fresh,
                             bool shown,
                             unsigned int episodes,
                             unsigned int runs,
                             unsigned int seed)
{
    corridor_runs result;
    const corridor_types::action right(corridor::go_right);
    for (unsigned int run = 0; run < runs; run++) {
        std::mt19937 prng(1000 * seed + run);
        corridor game;
        game.show_position(shown);
        learner_type learner = fresh;
        corridor_types::policy preferences;
        if (!shown) {
            const double half = std::log(19.0) / 2;
            preferences.update(corridor_types::state(0), right, -half);
            preferences.update(corridor_types::state(0), corridor_types::action(corridor::go_left), half);
        }
        // the agent draws its actions from π, the policy being learnt
        auto agent = [&](const corridor::state_type & s,
                         const std::vector<corridor::action_type> &) {
            return learner.sample(corridor_types::state(s), preferences, prng).trait();
        };
        for (unsigned int e = 0; e < episodes; e++) {
            learn_episode(game, agent, learner, preferences);
        }
        // π(right) in each cell, and the exact value of the policy learnt
        std::vector<double> p;
        for (unsigned int cell = 0; cell < 3; cell++) {
            p.push_back(learner.probability(corridor_types::state(shown ? cell : 0), right, preferences));
        }
        result.right.push_back(p[0]);
        result.values.push_back(shown ? corridor::start_value(p[0], p[1], p[2])
                                      : corridor::start_value(p[0]));
    }
    return result;
}

/// @return the median of @param values (the upper one of the two middle values, for an even count)
double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

/// @return how many of @param values are above @param threshold
std::size_t above(const std::vector<double> & values,
                  double threshold)
{
    return static_cast<std::size_t>(std::count_if(values.begin(), values.end(),
                                                  [threshold](double v) { return v > threshold; }));
}
} // namespace

/// verifies π(a | s), the softmax of the preferences
TEST_CASE("softmax_policy: the probabilities of the preferences", "[policy_gradient][softmax]")
{
    relearn::softmax_policy<state,action> pi(left_right);
    policy preferences;
    // a state never learnt: every preference is 0, so both actions are equally likely
    REQUIRE(pi.probability(state("s"), action("left"), preferences) == Approx(0.5));
    REQUIRE(pi.probability(state("s"), action("right"), preferences) == Approx(0.5));
    // h(right) = ln 3, h(left) = 0: π(right) = 3 / (3 + 1)
    preferences.update(state("s"), action("right"), std::log(3.0));
    REQUIRE(pi.probability(state("s"), action("right"), preferences) == Approx(0.75));
    REQUIRE(pi.probability(state("s"), action("left"), preferences) == Approx(0.25));
    auto all = pi.probabilities(state("s"), preferences);
    REQUIRE(all.size() == 2);
    REQUIRE(all[0].first == action("left"));
    REQUIRE(all[0].second + all[1].second == Approx(1));
    // an action that is not legal is never taken
    REQUIRE(pi.probability(state("s"), action("up"), preferences) == 0);
    // huge preferences do not overflow: e^1000 does not fit in a double, but only
    // their difference counts, π(right) = 1 / (1 + e^-1)
    preferences.update(state("t"), action("right"), 1000);
    preferences.update(state("t"), action("left"), 999);
    REQUIRE(pi.probability(state("t"), action("right"), preferences) == Approx(1 / (1 + std::exp(-1.0))));
    REQUIRE(pi.probability(state("t"), action("left"), preferences) == Approx(1 / (1 + std::exp(1.0))));
}

/// verifies ∇ ln π, and the step along it
TEST_CASE("softmax_policy: the gradient of ln π", "[policy_gradient][softmax]")
{
    relearn::softmax_policy<state,action> pi(left_right);
    policy preferences;
    preferences.update(state("s"), action("right"), std::log(3.0));
    // ∂ ln π(left | s) / ∂ h(s, b) = 1[left = b] - π(b | s): 1 - 0.25 for left, 0 - 0.75 for right
    auto gradient = pi.log_gradient(state("s"), action("left"), preferences);
    REQUIRE(gradient.size() == 2);
    REQUIRE(gradient[0].first == action("left"));
    REQUIRE(gradient[0].second == Approx(0.75));
    REQUIRE(gradient[1].second == Approx(-0.75));
    // a step of 2 along it: h(left) = 0 + 2 * 0.75, h(right) = ln 3 - 2 * 0.75
    pi.ascend(state("s"), action("left"), 2, preferences);
    REQUIRE(preferences.value(state("s"), action("left")) == Approx(1.5));
    REQUIRE(preferences.value(state("s"), action("right")) == Approx(std::log(3.0) - 1.5));
    // the preferences of other states do not move
    REQUIRE(preferences.actions(state("t")).empty());
}

/// verifies that the actions drawn follow π
TEST_CASE("softmax_policy: sample draws from π", "[policy_gradient][softmax]")
{
    relearn::softmax_policy<state,action> pi(left_right);
    policy preferences;
    preferences.update(state("s"), action("right"), std::log(3.0));
    std::mt19937 prng(1);
    // π(right) = 0.75: about 75000 of 100000 draws go right (the standard deviation is 137)
    int rights = 0;
    for (int i = 0; i < 100000; i++) {
        rights += pi.sample(state("s"), preferences, prng) == action("right");
    }
    REQUIRE(rights > 74000);
    REQUIRE(rights < 76000);
    // a huge preference: always that action
    preferences.update(state("t"), action("left"), 1000);
    for (int i = 0; i < 1000; i++) {
        REQUIRE(pi.sample(state("t"), preferences, prng) == action("left"));
    }
}

/// verifies one REINFORCE episode worked out by hand, with the γ^t factor
TEST_CASE("reinforce: one episode by hand", "[policy_gradient][reinforce]")
{
    // α = 0.5, γ = 0.5: (s0, right) earns 0, then (s1, left) earns 2.
    // The returns are G_1 = 2 and G_0 = 0 + 0.5 * 2 = 1, and π is 1/2 everywhere at first.
    //    t = 0: α γ^0 G_0 = 0.5, so h(s0, right) = 0.5 * (1 - 1/2) = 0.25, h(s0, left) = -0.25
    //    t = 1: α γ^1 G_1 = 0.5, so h(s1, left)  = 0.5 * (1 - 1/2) = 0.25, h(s1, right) = -0.25
    // (without γ^t, the step at t = 1 would be α G_1 = 1, and h(s1, left) 0.5)
    relearn::reinforce<state,action> learner(0.5, 0.5, left_right);
    policy preferences;
    learner(episode {{state(0, "s0"), action("right")},
                     {state(2, "s1"), action("left")}}, preferences);
    REQUIRE(preferences.value(state("s0"), action("right")) == Approx(0.25));
    REQUIRE(preferences.value(state("s0"), action("left")) == Approx(-0.25));
    REQUIRE(preferences.value(state("s1"), action("left")) == Approx(0.25));
    REQUIRE(preferences.value(state("s1"), action("right")) == Approx(-0.25));
    // the policy learnt: right is now more likely in s0, π(right | s0) = 1 / (1 + e^-0.5)
    REQUIRE(learner.probability(state("s0"), action("right"), preferences) == Approx(0.6224593));
}

/// verifies that θ changes from one step to the next whilst the episode is walked
TEST_CASE("reinforce: each step starts from the preferences left by the one before", "[policy_gradient][reinforce]")
{
    // α = 0.5, γ = 1: (s, right) twice, each earning -1. The returns are G_0 = -2 and G_1 = -1.
    //    t = 0: π(right) = 1/2, α G_0 = -1: h(right) = -1 * (1 - 1/2) = -0.5, h(left) = 0.5
    //    t = 1: π(right) = e^-0.5 / (e^-0.5 + e^0.5) = 1 / (1 + e) = 0.2689414, α G_1 = -0.5:
    //           h(right) = -0.5 - 0.5 * (1 - 0.2689414) = -0.8655293, h(left) = 0.8655293
    // (with π still at 1/2 for t = 1, h(right) would be -0.75)
    relearn::reinforce<state,action> learner(0.5, 1.0, left_right);
    policy preferences;
    learner(episode {{state(-1, "s"), action("right")},
                     {state(-1, "s"), action("right")}}, preferences);
    REQUIRE(preferences.value(state("s"), action("right")) == Approx(-0.8655293));
    REQUIRE(preferences.value(state("s"), action("left")) == Approx(0.8655293));
}

/// verifies that an episode cut short is not learnt
TEST_CASE("reinforce: an episode cut short is not learnt", "[policy_gradient][reinforce]")
{
    relearn::reinforce<state,action> plain(0.5, 1.0, left_right);
    relearn::reinforce_with_baseline<state,action> baseline(0.5, 0.5, 1.0, left_right);
    policy preferences;
    auto cut = episode {{state(-1, "s0"), action("right")},
                        {state(0, "s1"), action("left")}};
    plain(cut, preferences, false);
    baseline(cut, preferences, false);
    REQUIRE(preferences.actions(state("s0")).empty());
    REQUIRE(baseline.value(state("s0")) == 0);
}

/// verifies two episodes of REINFORCE with a baseline, worked out by hand
TEST_CASE("reinforce_with_baseline: two episodes by hand", "[policy_gradient][baseline]")
{
    // α^θ = 0.5, α^w = 0.5, γ = 0.5: (s0, right) earns 0, then (s1, left) earns 2, twice.
    // The returns are G_0 = 1 and G_1 = 2.
    relearn::reinforce_with_baseline<state,action> learner(0.5, 0.5, 0.5, left_right);
    policy preferences;
    auto twice = episode {{state(0, "s0"), action("right")},
                          {state(2, "s1"), action("left")}};
    // 1. t = 0: δ = 1 - 0 = 1, v(s0) = 0.5; α^θ γ^0 δ = 0.5, so h(s0, right) = 0.5 * (1 - 1/2) = 0.25
    //    t = 1: δ = 2 - 0 = 2, v(s1) = 1;   α^θ γ^1 δ = 0.5, so h(s1, left)  = 0.25
    learner(twice, preferences);
    REQUIRE(learner.value(state("s0")) == Approx(0.5));
    REQUIRE(learner.value(state("s1")) == Approx(1));
    REQUIRE(preferences.value(state("s0"), action("right")) == Approx(0.25));
    REQUIRE(preferences.value(state("s0"), action("left")) == Approx(-0.25));
    REQUIRE(preferences.value(state("s1"), action("left")) == Approx(0.25));
    // 2. now π(right | s0) = π(left | s1) = 1 / (1 + e^-0.5) = 0.6224593
    //    t = 0: δ = 1 - 0.5 = 0.5, v(s0) = 0.75; α^θ γ^0 δ = 0.25,
    //           so h(s0, right) = 0.25 + 0.25 * (1 - 0.6224593) = 0.3443852
    //    t = 1: δ = 2 - 1 = 1, v(s1) = 1.5;      α^θ γ^1 δ = 0.25, so h(s1, left) = 0.3443852
    learner(twice, preferences);
    REQUIRE(learner.value(state("s0")) == Approx(0.75));
    REQUIRE(learner.value(state("s1")) == Approx(1.5));
    REQUIRE(preferences.value(state("s0"), action("right")) == Approx(0.3443852));
    REQUIRE(preferences.value(state("s0"), action("left")) == Approx(-0.3443852));
    REQUIRE(preferences.value(state("s1"), action("left")) == Approx(0.3443852));
    REQUIRE(preferences.value(state("s1"), action("right")) == Approx(-0.3443852));
}

/// verifies two episodes of one-step Actor-Critic, worked out by hand, with I = γ^t
TEST_CASE("actor_critic: two episodes by hand", "[policy_gradient][actor_critic]")
{
    // α^θ = 0.5, α^w = 0.5, γ = 0.5: (s0, right) earns 1 and leads to s1, then (s1, left) earns 2 and ends.
    relearn::actor_critic<state,action> learner(0.5, 0.5, 0.5, left_right);
    policy preferences;
    auto twice = episode {{state(1, "s0"), action("right")},
                          {state(2, "s1"), action("left")}};
    // 1. t = 0: I = 1,   δ = 1 + 0.5 * v(s1) - v(s0) = 1, v(s0) = 0.5; α^θ I δ = 0.5, so h(s0, right) = 0.25
    //    t = 1: I = 0.5, δ = 2 - v(s1) = 2 (the end),      v(s1) = 1;   α^θ I δ = 0.5, so h(s1, left)  = 0.25
    REQUIRE(learner.discount() == 1);
    learner(twice, preferences);
    REQUIRE(learner.value(state("s0")) == Approx(0.5));
    REQUIRE(learner.value(state("s1")) == Approx(1));
    REQUIRE(preferences.value(state("s0"), action("right")) == Approx(0.25));
    REQUIRE(preferences.value(state("s0"), action("left")) == Approx(-0.25));
    REQUIRE(preferences.value(state("s1"), action("left")) == Approx(0.25));
    REQUIRE(preferences.value(state("s1"), action("right")) == Approx(-0.25));
    // the episode ended, so I is 1 again
    REQUIRE(learner.discount() == 1);
    // 2. now π(right | s0) = π(left | s1) = 0.6224593
    //    t = 0: δ = 1 + 0.5 * 1 - 0.5 = 1, v(s0) = 1;  α^θ I δ = 0.5,  so h(s0, right) = 0.25 + 0.5 * 0.3775407 = 0.4387704
    //    t = 1: δ = 2 - 1 = 1,             v(s1) = 1.5; α^θ I δ = 0.25, so h(s1, left)  = 0.25 + 0.25 * 0.3775407 = 0.3443852
    learner(twice, preferences);
    REQUIRE(learner.value(state("s0")) == Approx(1));
    REQUIRE(learner.value(state("s1")) == Approx(1.5));
    REQUIRE(preferences.value(state("s0"), action("right")) == Approx(0.4387704));
    REQUIRE(preferences.value(state("s1"), action("left")) == Approx(0.3443852));
}

/// verifies that learning step by step, as the episode is played, learns the same as the whole episode
TEST_CASE("actor_critic: step by step, and episodes cut short", "[policy_gradient][actor_critic]")
{
    // the same episode as above, learnt all at once by one learner, and step by step by another
    relearn::actor_critic<state,action> whole(0.5, 0.5, 0.5, left_right);
    relearn::actor_critic<state,action> stepwise(0.5, 0.5, 0.5, left_right);
    policy whole_preferences;
    policy stepwise_preferences;
    whole(episode {{state(1, "s0"), action("right")},
                   {state(2, "s1"), action("left")}}, whole_preferences);
    // step 0, learnt from the state reached (the action of the second link is not used)
    stepwise(episode {{state(1, "s0"), action("right")},
                      {state(0, "s1"), action("right")}}, stepwise_preferences, false);
    REQUIRE(stepwise.discount() == Approx(0.5));
    REQUIRE(stepwise.value(state("s1")) == 0);
    // step 1, the end of the episode
    stepwise(episode {{state(2, "s1"), action("left")}}, stepwise_preferences, true);
    REQUIRE(stepwise.discount() == 1);
    for (const auto & s : {"s0", "s1"}) {
        REQUIRE(stepwise.value(state(s)) == Approx(whole.value(state(s))));
        for (const auto & a : {"left", "right"}) {
            REQUIRE(stepwise_preferences.value(state(s), action(a)) ==
                    Approx(whole_preferences.value(state(s), action(a))));
        }
    }
    // an episode cut short leaves I where it was, until the next episode starts
    stepwise(episode {{state(1, "s0"), action("right")},
                      {state(0, "s1"), action("left")}}, stepwise_preferences, false);
    REQUIRE(stepwise.discount() == Approx(0.5));
    stepwise.start_episode();
    REQUIRE(stepwise.discount() == 1);
}

/// verifies the constructors, and brace initialisation
TEST_CASE("policy gradient: constructors and brace initialisation", "[policy_gradient]")
{
    relearn::reinforce<state,action> plain;
    REQUIRE(plain.alpha == Approx(0.01));
    REQUIRE(plain.gamma == Approx(0.9));
    REQUIRE_FALSE(plain.legal_actions);
    auto baseline = relearn::reinforce_with_baseline<state,action>{0.25, 0.5, 1.0, left_right};
    REQUIRE(baseline.alpha_theta == Approx(0.25));
    REQUIRE(baseline.alpha_w == Approx(0.5));
    REQUIRE(baseline.gamma == Approx(1));
    REQUIRE(baseline.legal_actions(state("s")).size() == 2);
    auto critic = relearn::actor_critic<state,action>{0.25, 0.5, 1.0, left_right};
    REQUIRE(critic.alpha_theta == Approx(0.25));
    REQUIRE(critic.alpha_w == Approx(0.5));
    REQUIRE(critic.discount() == 1);
    REQUIRE(critic.value(state("s")) == 0);
}

/// verifies Figure 13.1: REINFORCE learns close to the best stochastic policy of the short corridor
TEST_CASE("reinforce: the short corridor (Figure 13.1)", "[policy_gradient][reinforce][short_corridor]")
{
    // α = 2^-13 (the best step size of Figure 13.1), γ = 1, 50 runs of 1000 episodes, from
    // π(right) = 0.05, worth -82. The best policy is p = 0.5858, worth -11.657, and REINFORCE
    // comes to it from below. Over 10 seeds (the same numbers with libstdc++ and with libc++),
    // the median run learnt p = 0.470 to 0.486, worth -12.29 to -12.12, and every run but one
    // was worth more than -13.8. That one run (seed 10) had an unlucky 17th episode, 651 steps
    // mostly bumping left into the wall, whose large returns pushed p from 0.055 to 1 at once;
    // from there every episode is cut short at 1000 steps, and so never learnt
    corridor_reinforce learner(std::pow(2.0, -13), 1.0, corridor_actions);
    auto learnt = learn_corridor(learner, false, 1000, 50, 1);
    REQUIRE(median(learnt.right) > 0.42);
    REQUIRE(median(learnt.right) < 0.65);
    REQUIRE(median(learnt.values) > -12.7);
    REQUIRE(above(learnt.values, -14) >= 48);
}

/// verifies Figure 13.2: with a baseline, REINFORCE learns much faster
TEST_CASE("reinforce_with_baseline: the short corridor (Figure 13.2)", "[policy_gradient][baseline][short_corridor]")
{
    // α^θ = 2^-9 and α^w = 2^-6 with a baseline, against α = 2^-13 without (Figure 13.2),
    // γ = 1, the same 50 runs, from π(right) = 0.05. Over 10 seeds (the same numbers with
    // libstdc++ and with libc++), after 200 episodes the median run learnt a policy worth
    // -11.92 to -11.73 with the baseline, and -21.59 to -20.65 without. After 1000 episodes
    // with the baseline, the median run learnt p = 0.577 to 0.617, worth -11.84 to -11.73, and
    // at least 48 runs of 50 were worth more than -15 (the step size is large, so a few runs
    // wander off for a while)
    corridor_baseline baseline(std::pow(2.0, -9), std::pow(2.0, -6), 1.0, corridor_actions);
    corridor_reinforce plain(std::pow(2.0, -13), 1.0, corridor_actions);
    auto with = learn_corridor(baseline, false, 200, 50, 1);
    auto without = learn_corridor(plain, false, 200, 50, 1);
    REQUIRE(median(with.values) > -12.3);
    REQUIRE(median(without.values) < -19);
    // and it stays close to the best policy
    auto longer = learn_corridor(baseline, false, 1000, 50, 1);
    REQUIRE(median(longer.right) > 0.53);
    REQUIRE(median(longer.right) < 0.65);
    REQUIRE(median(longer.values) > -12.3);
    REQUIRE(above(longer.values, -15) >= 46);
}

/// verifies that Actor-Critic learns the best policy when it can tell the cells apart
TEST_CASE("actor_critic: the short corridor, with the cells shown", "[policy_gradient][actor_critic][short_corridor]")
{
    // the agent sees its cell, so the best policy is right, left, right, worth -3.
    // α^θ = 2^-4, α^w = 2^-3, γ = 1, 20 runs of 1000 episodes, from π(right) = 1/2 everywhere.
    // Over 10 seeds (the same numbers with libstdc++ and with libc++), every run learnt a policy
    // worth -3.039 to -3.014, with π(right) = 0.986 to 0.997 in the first cell
    corridor_critic learner(std::pow(2.0, -4), std::pow(2.0, -3), 1.0, corridor_actions);
    auto learnt = learn_corridor(learner, true, 1000, 20, 1);
    REQUIRE(above(learnt.values, -3.1) == learnt.values.size());
    REQUIRE(median(learnt.right) > 0.97);
}

/// verifies that Actor-Critic is misled by a critic which cannot tell the cells apart
TEST_CASE("actor_critic: the short corridor, with the cells looking the same", "[policy_gradient][actor_critic][short_corridor]")
{
    // The critic has one value v for all three cells, so the TD error of every step that does not
    // reach G is δ = -1 + v - v = -1, whichever action was taken: on average it does not move the
    // policy. Only the step onto G, right in the third cell, has δ = -1 - v, more than the -1 of
    // left there as soon as v < 0, and so right becomes more likely. p keeps growing, past the best
    // p = 0.5858, towards 1, where the agent goes back and forth around the second cell.
    // The step sizes of Figure 13.2, α^θ = 2^-9 and α^w = 2^-6, γ = 1, 5 runs of 1000 episodes,
    // from π(right) = 0.05. Over 10 seeds (the same numbers with libstdc++ and with libc++),
    // every run learnt p = 0.9981 to 0.9986, worth -1449 to -1074
    corridor_critic learner(std::pow(2.0, -9), std::pow(2.0, -6), 1.0, corridor_actions);
    auto learnt = learn_corridor(learner, false, 1000, 5, 1);
    for (std::size_t run = 0; run < learnt.right.size(); run++) {
        REQUIRE(learnt.right[run] > 0.99);
        REQUIRE(learnt.values[run] < -500);
    }
}
