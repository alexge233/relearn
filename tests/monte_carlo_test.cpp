// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/blackjack.hpp"
#include "solvers.hpp"
#include <cmath>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of Monte Carlo control (Sutton & Barto, Chapter 5):
 * - `monte_carlo`: returns worked out by hand, averaging, first-visit and every-visit
 * - `off_policy_monte_carlo`: importance weights worked out by hand
 * - Blackjack: Monte Carlo with exploring starts (Figure 5.2), on-policy with ε-greedy,
 *   and off-policy from random play, against the exact solution
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
