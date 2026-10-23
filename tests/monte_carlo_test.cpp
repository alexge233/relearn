// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/blackjack.hpp"
#include "../examples/games/infinite_variance.hpp"
#include "../examples/games/fixed_length.hpp"
#include "solvers.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of Monte Carlo control (Sutton & Barto, Chapter 5):
 * - `monte_carlo`: returns worked out by hand, averaging, first-visit and every-visit
 * - `off_policy_monte_carlo`: importance weights worked out by hand
 * - Blackjack: Monte Carlo with exploring starts (Figure 5.2), on-policy with ε-greedy,
 *   and off-policy from random play, against the exact solution
 * - `off_policy_monte_carlo_prediction`: the five importance samplings worked out by hand, against the book's
 *   equations written out here, unbiased (and their variances) by enumerating every episode of a short task,
 *   against plain Monte Carlo when π = b, Example 5.4 (Figure 5.3), Example 5.5's infinite variance, and long
 *   episodes with a reward at every step
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;

using game_type = games::blackjack;
using types     = games::types<game_type>;

/// @return b(a | s) = 1/2 for every action: the behaviour of a player who picks at random from two actions
double half(const state &,
            const action &)
{
    return 0.5;
}

/**
 * @brief how much less the greedy policy in @param memory earns than the optimal strategy
 * @param exact is the exact solution
 * @param prng breaks ties between actions of equal value
 * @return the optimal strategy's average reward, minus the learnt one's, over the same 200000 deals
 */
double shortfall(const types::policy & memory,
                 solvers::blackjack & exact,
                 std::mt19937 & prng)
{
    // the optimal strategy: the action of higher exact value
    auto optimal = [&](const game_type::state_type & s,
                       const std::vector<unsigned int> &) {
        double stick = s.natural ? exact.stick_natural(s.dealer_showing)
                                 : exact.stick(s.player_sum, s.dealer_showing);
        double hit = exact.hit(s.player_sum, s.usable_ace, s.dealer_showing);
        return stick >= hit ? game_type::stick : game_type::hit;
    };
    const unsigned int games_played = 200000;
    // both are dealt the same cards
    game_type eval_learnt(77);
    game_type eval_optimal(77);
    double learnt = games::evaluate(eval_learnt, memory, games_played, 100, prng);
    double best = 0;
    for (unsigned int i = 0; i < games_played; i++) {
        best += games::play(eval_optimal, optimal, 100).total_reward;
    }
    return best / games_played - learnt;
}

/**
 * @brief count the decision states where the learnt greedy action is not the optimal one
 * @param memory is the learnt policy
 * @param exact is the exact solution
 * @return the number of disagreements, of the 200 states of Figure 5.2
 */
int disagreements(const types::policy & memory,
                  solvers::blackjack & exact)
{
    int count = 0;
    for (int usable = 0; usable < 2; usable++) {
        for (unsigned int sum = 12; sum <= 21; sum++) {
            for (unsigned int dealer = 1; dealer <= 10; dealer++) {
                game_type::state_type s;
                s.player_sum     = sum;
                s.usable_ace     = usable == 1;
                s.dealer_showing = dealer;
                double q_stick = memory.value(types::state(s), types::action(game_type::stick));
                double q_hit   = memory.value(types::state(s), types::action(game_type::hit));
                bool best_sticks = exact.stick(sum, dealer) >= exact.hit(sum, usable == 1, dealer);
                count += (q_stick >= q_hit) != best_sticks;
            }
        }
    }
    return count;
}
} // namespace

/// verifies the returns, worked out backwards from the end of the episode
TEST_CASE("monte_carlo: each value is the return that followed", "[monte_carlo]")
{
    // γ = 0.5, rewards 0, 0, 1: the returns are 0.25, 0.5 and 1. The first return of a pair
    // is its average, so each value is exactly its return.
    relearn::monte_carlo<state,action> learner(0.5);
    policy memory;
    learner(episode {{state(0, "s0"), action("a")},
                     {state(0, "s1"), action("b")},
                     {state(1, "s2"), action("c")}}, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(0.25));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(0.5));
    REQUIRE(memory.value(state("s2"), action("c")) == Approx(1));
}

/// verifies that each value is the plain average of its returns
TEST_CASE("monte_carlo: values are averages of returns", "[monte_carlo]")
{
    // the same move, once followed by a win (1) and once by a loss (-1), then a draw (0)
    relearn::monte_carlo<state,action> learner(1.0);
    policy memory;
    learner(episode {{state(1, "s0"), action("a")}}, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(1));
    learner(episode {{state(-1, "s0"), action("a")}}, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(0));
    learner(episode {{state(0, "s0"), action("a")}}, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(0));
    REQUIRE(learner.visits.count(state("s0"), action("a")) == 3);

    // with a constant learning rate of 0.5 instead: 0.5, then 0.75
    relearn::monte_carlo<state,action> constant(1.0);
    constant.learning_rate = [](std::size_t) { return 0.5; };
    policy constant_memory;
    constant(episode {{state(1, "s0"), action("a")}}, constant_memory);
    REQUIRE(constant_memory.value(state("s0"), action("a")) == Approx(0.5));
    constant(episode {{state(1, "s0"), action("a")}}, constant_memory);
    REQUIRE(constant_memory.value(state("s0"), action("a")) == Approx(0.75));
}

/// verifies first-visit and every-visit Monte Carlo on an episode that visits a pair twice
TEST_CASE("monte_carlo: first-visit and every-visit", "[monte_carlo]")
{
    // γ = 1: (s0, a) earns 1, (s1, b) earns 0, then (s0, a) again earns 2.
    // The returns are 3 after the first visit of (s0, a), and 2 after the second.
    auto twice = episode {{state(1, "s0"), action("a")},
                          {state(0, "s1"), action("b")},
                          {state(2, "s0"), action("a")}};
    // first-visit: only the return after the first visit, 3
    relearn::monte_carlo<state,action> first(1.0, true);
    policy first_memory;
    first(twice, first_memory);
    REQUIRE(first_memory.value(state("s0"), action("a")) == Approx(3));
    REQUIRE(first.visits.count(state("s0"), action("a")) == 1);
    // every-visit: the average of both returns, (2 + 3) / 2
    relearn::monte_carlo<state,action> every(1.0, false);
    policy every_memory;
    every(twice, every_memory);
    REQUIRE(every_memory.value(state("s0"), action("a")) == Approx(2.5));
    REQUIRE(every.visits.count(state("s0"), action("a")) == 2);
}

/// verifies that an episode cut short is not learnt
TEST_CASE("monte_carlo: an episode cut short is not learnt", "[monte_carlo]")
{
    relearn::monte_carlo<state,action> on_policy(1.0);
    relearn::off_policy_monte_carlo<state,action> off_policy(1.0, half);
    policy memory;
    auto cut = episode {{state(1, "s0"), action("a")},
                        {state(0, "s1"), action("b")}};
    on_policy(cut, memory, false);
    off_policy(cut, memory, false);
    REQUIRE(memory.actions(state("s0")).empty());
    REQUIRE(on_policy.visits.count(state("s0"), action("a")) == 0);
}

/// verifies the importance weights of off-policy Monte Carlo, worked out by hand
TEST_CASE("off_policy_monte_carlo: weighted importance sampling", "[monte_carlo][off_policy]")
{
    // γ = 1, and the behaviour policy takes each action with probability 1/2.
    relearn::off_policy_monte_carlo<state,action> learner(1.0, half);
    policy memory;
    // 1. (s0, a) then (s1, b), which earns 1. Backwards:
    //    (s1, b): W = 1, C = 1, Q = 0 + 1/1 (1 - 0) = 1; b is greedy in s1, so W = 1 / (1/2) = 2
    //    (s0, a): C = 2, Q = 0 + 2/2 (1 - 0) = 1
    learner(episode {{state(0, "s0"), action("a")},
                     {state(1, "s1"), action("b")}}, memory);
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(1));
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(1));
    REQUIRE(learner.weights.value(state("s1"), action("b")) == Approx(1));
    REQUIRE(learner.weights.value(state("s0"), action("a")) == Approx(2));
    // 2. (s0, c) then (s1, b), which loses 1. Backwards:
    //    (s1, b): W = 1, C = 2, Q = 1 + 1/2 (-1 - 1) = 0; b is still greedy in s1, so W = 2
    //    (s0, c): C = 2, Q = 0 + 2/2 (-1 - 0) = -1
    learner(episode {{state(0, "s0"), action("c")},
                     {state(-1, "s1"), action("b")}}, memory);
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(0));
    REQUIRE(memory.value(state("s0"), action("c")) == Approx(-1));
    REQUIRE(learner.weights.value(state("s0"), action("c")) == Approx(2));
}

/// verifies that the walk back stops at the first move the greedy policy would not make
TEST_CASE("off_policy_monte_carlo: the earlier steps are not learnt after a non-greedy move", "[monte_carlo][off_policy]")
{
    // in s1 the greedy action is b (worth 10); the episode takes d there, so the greedy
    // policy would never have reached the end this way: (s1, d) is learnt, (s0, a) is not
    relearn::off_policy_monte_carlo<state,action> learner(1.0, half);
    policy memory;
    memory.update(state("s1"), action("b"), 10);
    learner(episode {{state(0, "s0"), action("a")},
                     {state(1, "s1"), action("d")}}, memory);
    REQUIRE(memory.value(state("s1"), action("d")) == Approx(1));
    REQUIRE(memory.actions(state("s0")).empty());
    REQUIRE(learner.weights.value(state("s0"), action("a")) == Approx(0));
}

/// verifies that b(a | s) is read before any value changes, when b is ε-greedy on the values being learnt
TEST_CASE("off_policy_monte_carlo: a behaviour policy that depends on Q", "[monte_carlo][off_policy]")
{
    // b is ε-greedy on memory, with ε = 0.1 and two actions: 0.95 for the greedy action, 0.05 for the other.
    // In s1 the greedy action is a (worth 1), but the episode explores x there, with probability 0.05.
    // Backwards, with γ = 1:
    //    (s1, x): W = 1, C = 1, Q = 0 + 1/1 (5 - 0) = 5, so x is now greedy; W = 1 / 0.05 = 20
    //    (s0, a): C = 0 + 20 = 20
    // Reading b after Q(s1, x) changed would give x 0.95 instead, and C(s0, a) = 1/0.95
    policy memory;
    memory.update(state("s1"), action("a"), 1);
    relearn::off_policy_monte_carlo<state,action> learner(1.0,
        [&](const state & s, const action & a) {
            auto greedy = memory.best_action(s);
            return (greedy && *greedy == a) ? 0.95 : 0.05;
        });
    learner(episode {{state(0, "s0"), action("a")},
                     {state(5, "s1"), action("x")}}, memory);
    REQUIRE(memory.value(state("s1"), action("x")) == Approx(5));
    REQUIRE(learner.weights.value(state("s0"), action("a")) == Approx(20));
}

/// verifies the constructors, and brace initialisation
TEST_CASE("monte_carlo: constructors and brace initialisation", "[monte_carlo]")
{
    relearn::monte_carlo<state,action> defaults;
    REQUIRE(defaults.gamma == Approx(0.9));
    REQUIRE(defaults.first_visit);
    auto every = relearn::monte_carlo<state,action>{1.0, false};
    REQUIRE(every.gamma == Approx(1));
    REQUIRE_FALSE(every.first_visit);
    auto off = relearn::off_policy_monte_carlo<state,action>{0.5, half};
    REQUIRE(off.gamma == Approx(0.5));
    REQUIRE(off.behaviour(state("s"), action("a")) == Approx(0.5));
}

/// verifies Figure 5.2: Monte Carlo with exploring starts learns the optimal strategy
TEST_CASE("monte_carlo: Blackjack with exploring starts (Figure 5.2)", "[monte_carlo][blackjack]")
{
    // every game starts from a random situation with a random first move, then plays greedily
    // (γ = 1, 500000 games). Over 10 seeds, with libstdc++ and with libc++, the learnt
    // strategy disagreed with the optimal one in 0 to 8 of the 200 states (near-ties), and
    // earned at most 0.0031 less than it
    game_type game(1);
    game.exploring_starts(true);
    std::mt19937 prng(2);
    types::policy memory;
    relearn::monte_carlo<types::state, types::action> learner(1.0);
    bool first = true;
    std::uniform_int_distribution<unsigned int> coin(0, 1);
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        // the first move is random, then greedy
        if (first) {
            first = false;
            return coin(prng);
        }
        return games::greedy<game_type>(memory, s, legal, prng);
    };
    for (unsigned int i = 0; i < 500000; i++) {
        first = true;
        auto r = games::play(game, agent, 100);
        learner(r.episode, memory, r.terminated);
    }
    solvers::blackjack exact;
    REQUIRE(disagreements(memory, exact) <= 15);
    REQUIRE(shortfall(memory, exact, prng) < 0.01);
}

/// verifies on-policy Monte Carlo control with ε-greedy exploration (Section 5.4)
TEST_CASE("monte_carlo: Blackjack on-policy, ε-greedy", "[monte_carlo][blackjack]")
{
    // ordinary deals, ε-greedy with ε = 0.1 (γ = 1, 500000 games). It learns the best
    // ε-soft policy, which is not quite the optimal one: over 10 seeds, with both standard
    // libraries, its greedy policy earned at most 0.0057 less than the optimal strategy
    game_type game(1);
    std::mt19937 prng(2);
    types::policy memory;
    relearn::monte_carlo<types::state, types::action> learner(1.0);
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
    };
    for (unsigned int i = 0; i < 500000; i++) {
        auto r = games::play(game, agent, 100);
        learner(r.episode, memory, r.terminated);
    }
    solvers::blackjack exact;
    REQUIRE(shortfall(memory, exact, prng) < 0.015);
}

/// verifies off-policy Monte Carlo control (Section 5.7): it learns the greedy policy from random play
TEST_CASE("off_policy_monte_carlo: Blackjack learnt from random play", "[monte_carlo][off_policy][blackjack]")
{
    // the player picks stick or hit at random, b = 1/2 (γ = 1, 500000 games). Over 10 seeds,
    // with both standard libraries, the greedy policy learnt disagreed with the optimal one
    // in 2 to 9 states, and earned at most 0.0027 less than it
    game_type game(1);
    std::mt19937 prng(2);
    types::policy memory;
    relearn::off_policy_monte_carlo<types::state, types::action> learner(1.0,
        [](const types::state &, const types::action &) { return 0.5; });
    std::uniform_int_distribution<unsigned int> coin(0, 1);
    auto random_player = [&](const game_type::state_type &,
                             const std::vector<unsigned int> &) {
        return coin(prng);
    };
    for (unsigned int i = 0; i < 500000; i++) {
        auto r = games::play(game, random_player, 100);
        learner(r.episode, memory, r.terminated);
    }
    solvers::blackjack exact;
    REQUIRE(disagreements(memory, exact) <= 15);
    REQUIRE(shortfall(memory, exact, prng) < 0.01);
}

namespace
{
using sampling_type = relearn::importance_sampling;
const std::vector<sampling_type> samplings = {sampling_type::ordinary, sampling_type::weighted,
                                              sampling_type::discounting_aware_ordinary,
                                              sampling_type::discounting_aware_weighted,
                                              sampling_type::per_decision};

/// @return the name of @param s, for the messages
std::string name_of(sampling_type s)
{
    switch (s) {
        case sampling_type::ordinary: return "ordinary";
        case sampling_type::weighted: return "weighted";
        case sampling_type::discounting_aware_ordinary: return "discounting-aware ordinary";
        case sampling_type::discounting_aware_weighted: return "discounting-aware weighted";
        default: return "per-decision";
    }
}

/// a probability of each state/action pair, for π and b in the tests
using probabilities = std::map<std::pair<std::string,std::string>,double>;

/**
 * @brief Sutton & Barto's estimators, written out here from their equations (5.5, 5.6, 5.9, 5.10 and 5.15),
 *        over all the episodes seen so far: the reference the learner is checked against
 * @param episodes are the episodes (each one ended), @param pi and @param b the two policies
 * @param gamma is γ, @param kind the importance sampling, @param first_visit first-visit or every-visit
 * @param s is the state, and @param a the action for an action value, or empty for the state value V(s)
 * @return the estimate: Σ numerators / Σ denominators, 0 while the denominators add up to 0
 */
double book_estimate(const std::vector<episode> & episodes,
                     const probabilities & pi,
                     const probabilities & b,
                     double gamma,
                     sampling_type kind,
                     bool first_visit,
                     const std::string & s,
                     const std::string & a)
{
    double numerator = 0;
    double denominator = 0;
    for (const auto & e : episodes) {
        const std::size_t T = e.size();
        // ρ_{i:k} = Π_{j=i}^{k} π(a_j | s_j) / b(a_j | s_j), 1 when k < i
        auto rho = [&](std::size_t i, std::size_t k) {
            double product = 1;
            for (std::size_t j = i; j <= k && j < T; j++) {
                auto pair = std::make_pair(e[j].state.trait(), e[j].action.trait());
                product *= pi.at(pair) / b.at(pair);
            }
            return product;
        };
        bool seen = false;
        for (std::size_t t = 0; t < T; t++) {
            if (e[t].state.trait() != s || (!a.empty() && e[t].action.trait() != a)) {
                continue;
            }
            if (first_visit && seen) {
                continue;
            }
            seen = true;
            // the first action whose ratio counts: a_t for V(s), a_{t+1} for Q(s, a)
            const std::size_t i = a.empty() ? t : t + 1;
            // the flat partial return Ḡ_{t:h} = r_t + ... + r_{h-1}, and the return G_t
            auto flat = [&](std::size_t h) {
                double sum = 0;
                for (std::size_t k = t; k < h; k++) {
                    sum += e[k].state.reward();
                }
                return sum;
            };
            double G = 0;
            for (std::size_t k = t; k < T; k++) {
                G += std::pow(gamma, static_cast<double>(k - t)) * e[k].state.reward();
            }
            const double whole = i <= T - 1 ? rho(i, T - 1) : 1.0;
            if (kind == sampling_type::ordinary) {
                numerator += whole * G;
                denominator += 1;
            }
            else if (kind == sampling_type::weighted) {
                numerator += whole * G;
                denominator += whole;
            }
            else if (kind == sampling_type::per_decision) {
                // G̃_t = Σ_k γ^{k-t} ρ_{i:k} r_k
                for (std::size_t k = t; k < T; k++) {
                    numerator += std::pow(gamma, static_cast<double>(k - t)) * (k >= i ? rho(i, k) : 1.0) * e[k].state.reward();
                }
                denominator += 1;
            }
            else {
                // (1 - γ) Σ_{h=t+1}^{T-1} γ^{h-t-1} ρ_{i:h-1} Ḡ_{t:h} + γ^{T-t-1} ρ_{i:T-1} Ḡ_{t:T}
                double sum = 0;
                double weights = 0;
                for (std::size_t h = t + 1; h < T; h++) {
                    const double w = (1 - gamma) * std::pow(gamma, static_cast<double>(h - t - 1)) * (h - 1 >= i ? rho(i, h - 1) : 1.0);
                    sum += w * flat(h);
                    weights += w;
                }
                const double w = std::pow(gamma, static_cast<double>(T - t - 1)) * whole;
                sum += w * flat(T);
                weights += w;
                numerator += sum;
                denominator += kind == sampling_type::discounting_aware_weighted ? weights : 1.0;
            }
        }
    }
    return denominator != 0 ? numerator / denominator : 0.0;
}

/// @return a function giving the probability of each pair in @param table
std::function<double(const state &, const action &)> lookup(const probabilities & table)
{
    return [table](const state & s, const action & a) { return table.at({s.trait(), a.trait()}); };
}
} // namespace

/// verifies the five importance samplings worked out by hand, every-visit and first-visit
TEST_CASE("off_policy_monte_carlo_prediction: one episode by hand", "[monte_carlo][prediction]")
{
    // π(a | x) = 0.8, π(b | x) = 0.2, π(a | y) = 0.4, π(b | y) = 0.6, b = 1/2 everywhere, γ = 0.5;
    // (x, a) earns 1, (y, b) 2, then (x, b) 4 and ends: the ratios are 1.6, 1.2 and 0.4.
    // The state x is visited twice: at t = 0, G_0 = 1 + 0.5 * 2 + 0.25 * 4 = 3 and ρ_{0:2} = 0.768, and at t = 2,
    // G_2 = 4 and ρ_{2:2} = 0.4. So, every-visit, ordinary V(x) = (0.768 * 3 + 0.4 * 4) / 2 = 1.952, and weighted
    // V(x) = 3.904 / 1.168 = 3.3425. The rest were worked out the same way (apart from relearn, in a few lines of Python)
    const probabilities pi = {{{"x", "a"}, 0.8}, {{"x", "b"}, 0.2}, {{"y", "a"}, 0.4}, {{"y", "b"}, 0.6}};
    auto once = episode {{state(1, "x"), action("a")},
                         {state(2, "y"), action("b")},
                         {state(4, "x"), action("b")}};
    struct expected
    {
        sampling_type sampling;
        bool first_visit;
        double v_x, v_y, q_xa, q_yb, q_xb;
    };
    const std::vector<expected> table = {
        {sampling_type::ordinary, false, 1.952, 1.92, 1.44, 1.6, 4},
        {sampling_type::ordinary, true, 2.304, 1.92, 1.44, 1.6, 4},
        {sampling_type::weighted, false, 3.3424657534246576, 4, 3, 4, 4},
        {sampling_type::weighted, true, 3, 4, 3, 4, 4},
        {sampling_type::discounting_aware_ordinary, false, 2.592, 2.64, 2.24, 2.2, 4},
        {sampling_type::discounting_aware_ordinary, true, 3.584, 2.64, 2.24, 2.2, 4},
        {sampling_type::discounting_aware_weighted, false, 2.76923076923077, 3.142857142857143, 2.434782608695652, 3.142857142857143, 4},
        {sampling_type::discounting_aware_weighted, true, 2.4347826086956528, 3.142857142857143, 2.434782608695652, 3.142857142857143, 4},
        {sampling_type::per_decision, false, 2.944, 3.36, 2.68, 2.8, 4},
        {sampling_type::per_decision, true, 4.288, 3.36, 2.68, 2.8, 4},
    };
    for (const auto & row : table) {
        INFO(name_of(row.sampling) << (row.first_visit ? ", first-visit" : ", every-visit"));
        relearn::off_policy_monte_carlo_prediction<state,action> learner(0.5, lookup(pi), half, row.sampling, row.first_visit);
        policy q;
        learner(once, q);
        REQUIRE(learner.value(state("x")) == Approx(row.v_x));
        REQUIRE(learner.value(state("y")) == Approx(row.v_y));
        REQUIRE(q.value(state("x"), action("a")) == Approx(row.q_xa));
        REQUIRE(q.value(state("y"), action("b")) == Approx(row.q_yb));
        REQUIRE(q.value(state("x"), action("b")) == Approx(row.q_xb));
    }
}

/// verifies that ordinary importance sampling counts the episodes π would never have played, and weighted does not
TEST_CASE("off_policy_monte_carlo_prediction: a ratio of 0, and an episode cut short", "[monte_carlo][prediction]")
{
    // π always takes a, b = 1/2: (s, a) earning 2 has ρ = 2, so ordinary V(s) = 4 and weighted V(s) = 2.
    // Then (s, b) earning 10, which π never takes: ρ = 0. Ordinary averages it in as a 0, V(s) = (4 + 0) / 2 = 2;
    // weighted adds 0 to both sums, and stays at 2
    const probabilities pi = {{{"s", "a"}, 1.0}, {{"s", "b"}, 0.0}};
    relearn::off_policy_monte_carlo_prediction<state,action> ordinary(1.0, lookup(pi), half, sampling_type::ordinary);
    relearn::off_policy_monte_carlo_prediction<state,action> weighted(1.0, lookup(pi), half, sampling_type::weighted);
    policy q_ordinary, q_weighted;
    for (auto * learner : {&ordinary, &weighted}) {
        (*learner)(episode {{state(2, "s"), action("a")}}, learner == &ordinary ? q_ordinary : q_weighted);
    }
    REQUIRE(ordinary.value(state("s")) == Approx(4));
    REQUIRE(weighted.value(state("s")) == Approx(2));
    for (auto * learner : {&ordinary, &weighted}) {
        (*learner)(episode {{state(10, "s"), action("b")}}, learner == &ordinary ? q_ordinary : q_weighted);
    }
    REQUIRE(ordinary.value(state("s")) == Approx(2));
    REQUIRE(weighted.value(state("s")) == Approx(2));
    // the action value of (s, b) needs no ratio at all (b is given, and nothing follows): it is its return, 10
    REQUIRE(q_ordinary.value(state("s"), action("b")) == Approx(10));
    REQUIRE(q_weighted.value(state("s"), action("b")) == Approx(10));
    // a state never visited, or whose weights are all 0, is worth 0
    REQUIRE(weighted.value(state("never")) == 0);
    relearn::off_policy_monte_carlo_prediction<state,action> zero(1.0, lookup(pi), half, sampling_type::weighted);
    policy q_zero;
    zero(episode {{state(10, "s"), action("b")}}, q_zero);
    REQUIRE(zero.value(state("s")) == 0);
    // and so is an action value: (s, a) followed by (s, b), which π never takes, gives (s, a) a weight of 0
    relearn::off_policy_monte_carlo_prediction<state,action> both(1.0, lookup(pi), half, sampling_type::weighted);
    policy q_both;
    both(episode {{state(2, "s"), action("a")}, {state(10, "s"), action("b")}}, q_both);
    REQUIRE(q_both.value(state("s"), action("a")) == 0);
    REQUIRE(q_both.value(state("s"), action("b")) == Approx(10));
    // an episode cut short is not learnt at all
    zero(episode {{state(5, "s"), action("a")}}, q_zero, false);
    REQUIRE(zero.value(state("s")) == 0);
    REQUIRE(q_zero.value(state("s"), action("a")) == 0);
}

/// verifies the learner against the book's equations, written out here, after every episode
TEST_CASE("off_policy_monte_carlo_prediction: the book's equations, episode by episode", "[monte_carlo][prediction]")
{
    // 100 runs of 10 random episodes over the states s0 ... s3 (which come back often) and the actions a and b,
    // with random π and b, and γ random or 1: after each episode, every state value and action value of the learner
    // is the estimate of equations 5.5, 5.6, 5.9, 5.10 and 5.15 worked out over all the episodes so far
    std::mt19937 prng(3);
    std::uniform_int_distribution<int> pick_state(0, 3);
    std::uniform_int_distribution<int> pick_action(0, 1);
    std::uniform_int_distribution<int> pick_reward(-2, 3);
    std::uniform_int_distribution<int> pick_length(1, 8);
    std::uniform_real_distribution<double> uniform(0.05, 0.95);
    double largest = 0;
    // std::max would drop a NaN, so a difference that isn't a number counts as an infinite one
    auto worst = [&largest](double difference) {
        largest = std::isfinite(difference) ? std::max(largest, std::fabs(difference))
                                            : std::numeric_limits<double>::infinity();
    };
    for (int run = 0; run < 100; run++) {
        probabilities pi, b;
        for (int k = 0; k < 4; k++) {
            const std::string s = "s" + std::to_string(k);
            // π may give an action no chance at all, b never does
            const double p = run % 3 == 0 ? (k % 2 == 0 ? 1.0 : 0.0) : uniform(prng);
            const double q = uniform(prng);
            pi[{s, "a"}] = p;
            pi[{s, "b"}] = 1 - p;
            b[{s, "a"}] = q;
            b[{s, "b"}] = 1 - q;
        }
        const double gamma = run % 4 == 0 ? 1.0 : uniform(prng);
        std::vector<episode> episodes;
        for (int e = 0; e < 10; e++) {
            episode one;
            const int length = pick_length(prng);
            for (int i = 0; i < length; i++) {
                one.push_back({state(pick_reward(prng), "s" + std::to_string(pick_state(prng))),
                               action(pick_action(prng) == 0 ? "a" : "b")});
            }
            episodes.push_back(one);
        }
        for (auto kind : samplings) {
            for (bool first_visit : {false, true}) {
                relearn::off_policy_monte_carlo_prediction<state,action> learner(gamma, lookup(pi), lookup(b), kind, first_visit);
                policy q;
                std::vector<episode> so_far;
                for (const auto & e : episodes) {
                    learner(e, q);
                    so_far.push_back(e);
                    for (int k = 0; k < 4; k++) {
                        const std::string s = "s" + std::to_string(k);
                        worst(learner.value(state(s)) - book_estimate(so_far, pi, b, gamma, kind, first_visit, s, ""));
                        for (const auto & a : {"a", "b"}) {
                            worst(q.value(state(s), action(a)) - book_estimate(so_far, pi, b, gamma, kind, first_visit, s, a));
                        }
                    }
                }
            }
        }
    }
    // the largest difference was 4.5e-13, on values made of ratios up to (1 / 0.05)^8
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 1e-10);
}

/// verifies, by enumerating every episode of a short task, that ordinary, discounting-aware ordinary and per-decision
/// importance sampling are unbiased, and that the last two have a smaller variance
TEST_CASE("off_policy_monte_carlo_prediction: unbiased, and the variances, over every episode of a short task", "[monte_carlo][prediction]")
{
    // games::fixed_length with 4 steps: every episode is 4 actions, 0 (earns 1) or 1 (earns 0), 16 episodes in all.
    // π takes 0 with 0.9, b with 0.5, and γ = 0.5. For each episode, the estimate after that one episode alone is worked
    // out by the learner, and weighted by the episode's probability under b: the mean is then exact, as is the variance.
    games::fixed_length task(4);
    const double gamma = 0.5;
    using fixed_types = games::types<games::fixed_length>;
    auto pi = [](const fixed_types::state &, const fixed_types::action & a) { return a.trait() == 0 ? 0.9 : 0.1; };
    auto b = [](const fixed_types::state &, const fixed_types::action &) { return 0.5; };
    const double v = task.value(0, 0.9, gamma);
    const double q0 = 1 + gamma * task.value(1, 0.9, gamma);
    std::map<sampling_type,double> variance;
    for (auto kind : samplings) {
        double mean_v = 0, square_v = 0, mean_q = 0;
        for (unsigned int bits = 0; bits < 16; bits++) {
            fixed_types::episode e;
            for (unsigned int t = 0; t < 4; t++) {
                const unsigned int a = (bits >> t) & 1u;
                e.push_back({fixed_types::state(a == 0 ? 1.0 : 0.0, t), fixed_types::action(a)});
            }
            relearn::off_policy_monte_carlo_prediction<fixed_types::state, fixed_types::action> learner(gamma, pi, b, kind);
            fixed_types::policy q;
            learner(e, q);
            const double estimate = learner.value(fixed_types::state(0));
            mean_v += estimate / 16;
            square_v += estimate * estimate / 16;
            // q(0, 0): the episodes that start with action 0, each with probability 1/8 given that start
            if ((bits & 1u) == 0) {
                mean_q += q.value(fixed_types::state(0), fixed_types::action(0)) / 8;
            }
        }
        variance[kind] = square_v - mean_v * mean_v;
        INFO(name_of(kind) << ": mean " << mean_v << " (v = " << v << "), variance " << variance[kind]);
        if (kind == sampling_type::weighted || kind == sampling_type::discounting_aware_weighted) {
            // after a single episode, the weighted estimate is simply a return: biased
            REQUIRE(std::fabs(mean_v - v) > 0.01);
        }
        else {
            REQUIRE(mean_v == Approx(v).epsilon(1e-12));
            REQUIRE(mean_q == Approx(q0).epsilon(1e-12));
        }
    }
    // the variances of a single episode's estimate: discounting-aware and per-decision are well below ordinary
    INFO("ordinary " << variance[sampling_type::ordinary] << ", discounting-aware " << variance[sampling_type::discounting_aware_ordinary]
         << ", per-decision " << variance[sampling_type::per_decision]);
    REQUIRE(variance[sampling_type::discounting_aware_ordinary] < 0.5 * variance[sampling_type::ordinary]);
    REQUIRE(variance[sampling_type::per_decision] < 0.5 * variance[sampling_type::ordinary]);
}

/// verifies the special cases: on-policy it is plain Monte Carlo, γ = 1 removes discounting-awareness,
/// and a reward only at the end makes per-decision ordinary
TEST_CASE("off_policy_monte_carlo_prediction: on-policy it is monte_carlo, and the samplings that coincide", "[monte_carlo][prediction]")
{
    std::mt19937 prng(4);
    std::uniform_int_distribution<int> pick_state(0, 3);
    std::uniform_int_distribution<int> pick_action(0, 1);
    std::uniform_int_distribution<int> pick_reward(-2, 3);
    std::uniform_int_distribution<int> pick_length(1, 8);
    std::vector<episode> episodes, end_only;
    for (int e = 0; e < 300; e++) {
        episode one, last;
        const int length = pick_length(prng);
        for (int i = 0; i < length; i++) {
            const std::string s = "s" + std::to_string(pick_state(prng));
            const std::string a = pick_action(prng) == 0 ? "a" : "b";
            const int r = pick_reward(prng);
            one.push_back({state(r, s), action(a)});
            last.push_back({state(i + 1 == length ? r : 0, s), action(a)});
        }
        episodes.push_back(one);
        end_only.push_back(last);
    }
    const probabilities same = {{{"s0", "a"}, 0.3}, {{"s0", "b"}, 0.7}, {{"s1", "a"}, 0.5}, {{"s1", "b"}, 0.5},
                                {{"s2", "a"}, 0.9}, {{"s2", "b"}, 0.1}, {{"s3", "a"}, 0.6}, {{"s3", "b"}, 0.4}};
    const probabilities other = {{{"s0", "a"}, 0.8}, {{"s0", "b"}, 0.2}, {{"s1", "a"}, 0.1}, {{"s1", "b"}, 0.9},
                                 {{"s2", "a"}, 0.4}, {{"s2", "b"}, 0.6}, {{"s3", "a"}, 0.5}, {{"s3", "b"}, 0.5}};
    // π = b: every ratio is 1, and all five give monte_carlo's averages (every-visit and first-visit), to within 1e-12
    for (bool first_visit : {false, true}) {
        relearn::monte_carlo<state,action> plain(0.8, first_visit);
        policy plain_q;
        for (const auto & e : episodes) {
            plain(e, plain_q);
        }
        for (auto kind : samplings) {
            relearn::off_policy_monte_carlo_prediction<state,action> learner(0.8, lookup(same), lookup(same), kind, first_visit);
            policy q;
            for (const auto & e : episodes) {
                learner(e, q);
            }
            for (int k = 0; k < 4; k++) {
                for (const auto & a : {"a", "b"}) {
                    const state s("s" + std::to_string(k));
                    INFO(name_of(kind) << (first_visit ? ", first-visit, " : ", every-visit, ") << s.trait() << ", " << a);
                    // with every ratio 1, the flat partial returns, and the per-decision rewards, add up to G_t
                    REQUIRE(q.value(s, action(a)) == Approx(plain_q.value(s, action(a))).epsilon(1e-12).margin(1e-12));
                }
            }
        }
    }
    // γ = 1: discounting-aware ordinary and weighted are ordinary and weighted; and with a reward only at the end of each
    // episode, per-decision is ordinary (to within 1e-12)
    auto learn = [&](const std::vector<episode> & es, double gamma, sampling_type kind, policy & q) {
        relearn::off_policy_monte_carlo_prediction<state,action> learner(gamma, lookup(other), lookup(same), kind);
        for (const auto & e : es) {
            learner(e, q);
        }
        std::vector<double> v;
        for (int k = 0; k < 4; k++) {
            v.push_back(learner.value(state("s" + std::to_string(k))));
        }
        return v;
    };
    const std::vector<std::pair<sampling_type,sampling_type>> pairs = {
        {sampling_type::discounting_aware_ordinary, sampling_type::ordinary},
        {sampling_type::discounting_aware_weighted, sampling_type::weighted}};
    for (const auto & pair : pairs) {
        policy q1, q2;
        auto v1 = learn(episodes, 1.0, pair.first, q1);
        auto v2 = learn(episodes, 1.0, pair.second, q2);
        for (int k = 0; k < 4; k++) {
            INFO(name_of(pair.first) << ", s" << k);
            REQUIRE(v1[static_cast<std::size_t>(k)] == Approx(v2[static_cast<std::size_t>(k)]).epsilon(1e-12).margin(1e-12));
            for (const auto & a : {"a", "b"}) {
                const state s("s" + std::to_string(k));
                REQUIRE(q1.value(s, action(a)) == Approx(q2.value(s, action(a))).epsilon(1e-12).margin(1e-12));
            }
        }
    }
    for (double gamma : {1.0, 0.7}) {
        policy q1, q2;
        auto v1 = learn(end_only, gamma, sampling_type::per_decision, q1);
        auto v2 = learn(end_only, gamma, sampling_type::ordinary, q2);
        for (int k = 0; k < 4; k++) {
            INFO("γ = " << gamma << ", s" << k);
            REQUIRE(v1[static_cast<std::size_t>(k)] == Approx(v2[static_cast<std::size_t>(k)]).epsilon(1e-12).margin(1e-12));
        }
    }
}

/// verifies the constructor of the prediction learner, and brace initialisation
TEST_CASE("off_policy_monte_carlo_prediction: constructors and brace initialisation", "[monte_carlo][prediction]")
{
    relearn::off_policy_monte_carlo_prediction<state,action> defaults;
    REQUIRE(defaults.gamma == Approx(0.9));
    REQUIRE_FALSE(defaults.target);
    REQUIRE_FALSE(defaults.behaviour);
    REQUIRE(defaults.sampling == sampling_type::weighted);
    REQUIRE_FALSE(defaults.first_visit);
    auto braced = relearn::off_policy_monte_carlo_prediction<state,action>{1.0, half, half, sampling_type::per_decision, true};
    REQUIRE(braced.gamma == Approx(1));
    REQUIRE(braced.target(state("s"), action("a")) == Approx(0.5));
    REQUIRE(braced.behaviour(state("s"), action("a")) == Approx(0.5));
    REQUIRE(braced.sampling == sampling_type::per_decision);
    REQUIRE(braced.first_visit);
}

namespace
{
/**
 * @brief Example 5.4: the mean squared error of V(s), over @param runs runs, after 1, 10, 100, 1000 and 10000 episodes
 * @param kind is the importance sampling, @param seed seeds the runs (seed * 1000 + run)
 * @return the mean squared error, against v_π(s) = -0.27726, after each of those numbers of episodes
 *
 * Every game starts with the dealer showing a deuce and the player on 13 with a usable Ace, the player picks
 * stick or hit at random (b = 1/2), and the target policy sticks only on 20 or 21; γ = 1.
 */
std::vector<double> figure_5_3(sampling_type kind,
                               unsigned int runs,
                               unsigned int seed)
{
    const double truth = -0.27726;
    auto pi = [](const types::state & s, const types::action & a) {
        const bool stick = s.trait().player_sum >= 20;
        return (a.trait() == game_type::stick) == stick ? 1.0 : 0.0;
    };
    auto b = [](const types::state &, const types::action &) { return 0.5; };
    const std::vector<unsigned int> marks = {1, 10, 100, 1000, 10000};
    std::vector<double> error(marks.size(), 0);
    for (unsigned int run = 0; run < runs; run++) {
        game_type game(seed * 1000 + run);
        game.fixed_start(13, true, 2);
        std::mt19937 prng(seed * 1000 + run + 500);
        std::uniform_int_distribution<unsigned int> coin(0, 1);
        auto random_player = [&](const game_type::state_type &, const std::vector<unsigned int> &) { return coin(prng); };
        relearn::off_policy_monte_carlo_prediction<types::state, types::action> learner(1.0, pi, b, kind);
        types::policy q;
        std::size_t m = 0;
        // the state of Example 5.4, where every game starts
        game_type::state_type start;
        start.player_sum = 13;
        start.usable_ace = true;
        start.dealer_showing = 2;
        for (unsigned int e = 1; e <= marks.back(); e++) {
            auto r = games::play(game, random_player, 100);
            learner(r.episode, q, r.terminated);
            if (e == marks[m]) {
                const double v = learner.value(types::state(start));
                error[m] += (v - truth) * (v - truth) / runs;
                m++;
            }
        }
    }
    return error;
}
} // namespace

/// verifies the blackjack state of Example 5.4: v_π(s) = -0.27726, and Figure 5.3
TEST_CASE("off_policy_monte_carlo_prediction: Example 5.4, a blackjack state (Figure 5.3)", "[monte_carlo][prediction][blackjack]")
{
    // first, on-policy, the true value: 1000000 games played by π itself. Over 10 seeds, with libstdc++ and with libc++,
    // their average was -0.27843 to -0.27556
    game_type game(9);
    game.fixed_start(13, true, 2);
    double total = 0;
    for (unsigned int i = 0; i < 1000000; i++) {
        auto s = game.reset();
        while (true) {
            auto out = game.step(s.player_sum >= 20 ? game_type::stick : game_type::hit);
            total += out.reward;
            if (out.terminated) {
                break;
            }
            s = out.next;
        }
    }
    REQUIRE(total / 1000000 == Approx(-0.27726).margin(0.003));
    // then from random play: 100 runs of 10000 games. Over 10 seeds, with libstdc++ and with libc++, the mean squared
    // errors after 1 game were 4.1 to 21.7 (ordinary) against 0.13 to 0.26 (weighted), after 10 games 0.60 to 1.88 against
    // 0.43 to 0.58, and after 10000 games 0.00066 to 0.00115 for both. After 10 games weighted was still better with
    // the same games, but by as little as 15%, and after 100 games they overlap (0.073 to 0.13 against 0.068 to 0.12),
    // so the test compares them after 1 game only
    auto ordinary = figure_5_3(sampling_type::ordinary, 100, 1);
    auto weighted = figure_5_3(sampling_type::weighted, 100, 1);
    INFO("after 1, 10, 100, 1000, 10000 games: ordinary " << ordinary[0] << ", " << ordinary[1] << ", " << ordinary[2]
         << ", " << ordinary[3] << ", " << ordinary[4] << "; weighted " << weighted[0] << ", " << weighted[1] << ", "
         << weighted[2] << ", " << weighted[3] << ", " << weighted[4]);
    REQUIRE(weighted[0] < ordinary[0]);
    REQUIRE(ordinary[4] < 0.002);
    REQUIRE(weighted[4] < 0.002);
}

/// verifies Example 5.5: ordinary importance sampling has an infinite variance there, weighted gets the value at once
TEST_CASE("off_policy_monte_carlo_prediction: Example 5.5, the infinite variance", "[monte_carlo][prediction]")
{
    // π always goes left, b goes left and right with 1/2 each, γ = 1, first-visit: v_π(s) = 1. 10 runs of 10000 episodes
    using loop_types = games::types<games::infinite_variance>;
    auto pi = [](const loop_types::state &, const loop_types::action & a) {
        return a.trait() == games::infinite_variance::left ? 1.0 : 0.0;
    };
    auto b = [](const loop_types::state &, const loop_types::action &) { return 0.5; };
    std::vector<double> ordinary_estimates;
    for (unsigned int run = 0; run < 10; run++) {
        games::infinite_variance ordinary_game(run), weighted_game(run);
        std::mt19937 ordinary_prng(100 + run), weighted_prng(100 + run);
        std::uniform_int_distribution<unsigned int> coin(0, 1);
        relearn::off_policy_monte_carlo_prediction<loop_types::state, loop_types::action> ordinary(1.0, pi, b, sampling_type::ordinary, true);
        relearn::off_policy_monte_carlo_prediction<loop_types::state, loop_types::action> weighted(1.0, pi, b, sampling_type::weighted, true);
        loop_types::policy q_ordinary, q_weighted;
        bool seen = false;
        for (unsigned int e = 0; e < 10000; e++) {
            auto r = games::play(ordinary_game, [&](const unsigned int &, const std::vector<unsigned int> &) { return coin(ordinary_prng); }, 100000);
            ordinary(r.episode, q_ordinary, r.terminated);
            auto w = games::play(weighted_game, [&](const unsigned int &, const std::vector<unsigned int> &) { return coin(weighted_prng); }, 100000);
            weighted(w.episode, q_weighted, w.terminated);
            // once an episode π could have played (left all along, then the end) has been seen, weighted is exactly 1
            bool all_left = true;
            for (const auto & link : w.episode) {
                all_left = all_left && link.action.trait() == games::infinite_variance::left;
            }
            seen = seen || all_left;
            REQUIRE(weighted.value(loop_types::state(0)) == (seen ? 1.0 : 0.0));
        }
        ordinary_estimates.push_back(ordinary.value(loop_types::state(0)));
    }
    // ordinary importance sampling is unbiased, but after 10000 episodes its estimates are still all over the place:
    // over 10 seeds, with libstdc++ and with libc++, the lowest of the 10 runs was 0.45 to 0.59, the highest 0.96 to 2.49,
    // and they were at least 0.37 apart
    const auto lowest = *std::min_element(ordinary_estimates.begin(), ordinary_estimates.end());
    const auto highest = *std::max_element(ordinary_estimates.begin(), ordinary_estimates.end());
    INFO("ordinary importance sampling, after 10000 episodes: " << lowest << " to " << highest);
    REQUIRE(highest - lowest > 0.3);
}

/// verifies long episodes with a reward at every step (Sections 5.8 and 5.9): discounting-aware and per-decision win
TEST_CASE("off_policy_monte_carlo_prediction: long episodes with a reward at every step", "[monte_carlo][prediction]")
{
    // games::fixed_length with 20 steps, π takes action 0 (worth 1) with 0.9, b with 0.5, γ = 0.5. 100 runs of 1000
    // episodes: the mean squared error of V(0) against v_π(0) = 0.9 (1 - 0.5^20) / 0.5. Over 10 seeds, with libstdc++ and
    // with libc++, it was 2.1 to 673 for ordinary importance sampling, 0.053 to 0.103 for weighted, 0.0074 to 0.0116 for
    // discounting-aware ordinary, 0.0017 to 0.0030 for discounting-aware weighted, and 0.0034 to 0.0055 for per-decision
    using fixed_types = games::types<games::fixed_length>;
    games::fixed_length task(20);
    const double gamma = 0.5;
    const double v = task.value(0, 0.9, gamma);
    auto pi = [](const fixed_types::state &, const fixed_types::action & a) { return a.trait() == 0 ? 0.9 : 0.1; };
    auto b = [](const fixed_types::state &, const fixed_types::action &) { return 0.5; };
    std::map<sampling_type,double> error;
    for (auto kind : samplings) {
        for (unsigned int run = 0; run < 100; run++) {
            std::mt19937 prng(1000 + run);
            std::uniform_int_distribution<unsigned int> coin(0, 1);
            relearn::off_policy_monte_carlo_prediction<fixed_types::state, fixed_types::action> learner(gamma, pi, b, kind);
            fixed_types::policy q;
            for (unsigned int e = 0; e < 1000; e++) {
                auto r = games::play(task, [&](const unsigned int &, const std::vector<unsigned int> &) { return coin(prng); }, 100);
                learner(r.episode, q, r.terminated);
            }
            const double estimate = learner.value(fixed_types::state(0));
            error[kind] += (estimate - v) * (estimate - v) / 100;
        }
    }
    INFO("ordinary " << error[sampling_type::ordinary] << ", weighted " << error[sampling_type::weighted]
         << ", discounting-aware ordinary " << error[sampling_type::discounting_aware_ordinary]
         << ", discounting-aware weighted " << error[sampling_type::discounting_aware_weighted]
         << ", per-decision " << error[sampling_type::per_decision]);
    REQUIRE(error[sampling_type::discounting_aware_ordinary] < 0.01 * error[sampling_type::ordinary]);
    REQUIRE(error[sampling_type::per_decision] < 0.01 * error[sampling_type::ordinary]);
    REQUIRE(error[sampling_type::discounting_aware_weighted] < error[sampling_type::weighted]);
    REQUIRE(error[sampling_type::per_decision] < 0.01);
    REQUIRE(error[sampling_type::discounting_aware_weighted] < 0.005);
}
