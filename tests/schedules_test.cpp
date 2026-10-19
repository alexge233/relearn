// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include <cmath>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of the schedules (`polynomial_decay`, `exponential_decay`, `linear_decay`),
 * of `visit_counter`, and of the learners' per-visit learning rate `learning_rate`.
 * Every value is worked out by hand.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;

/**
 * @brief a one-step episode: (s, a) earns @param reward, and the episode ends
 * @return the episode
 */
episode one_step(double reward)
{
    return episode {{state(reward, "s"), action("a")}};
}

/**
 * @brief learn the rewards 1, 2, 3 and 4, one per episode, with α = 1/n per visit
 * @param learner is the learner to train
 * @return the learnt value of (s, a)
 */
template <class learner_type>
double average_of_four(learner_type & learner)
{
    policy memory;
    learner.learning_rate = relearn::polynomial_decay<>(1.0, 1.0);
    for (double r = 1; r <= 4; r++) {
        learner(one_step(r), memory);
    }
    return memory.value(state("s"), action("a"));
}
} // namespace

/// verifies polynomial_decay: v(k) = initial / k^exponent
TEST_CASE("polynomial_decay: initial / k^exponent", "[schedules]")
{
    // the default is 1/k
    relearn::polynomial_decay<> harmonic;
    REQUIRE(harmonic(1) == Approx(1.0));
    REQUIRE(harmonic(2) == Approx(0.5));
    REQUIRE(harmonic(4) == Approx(0.25));

    // 0.5 / √k
    relearn::polynomial_decay<> slow(0.5, 0.5);
    REQUIRE(slow(1) == Approx(0.5));
    REQUIRE(slow(4) == Approx(0.25));
    REQUIRE(slow(16) == Approx(0.125));

    // k = 0 counts as k = 1, so there is no division by zero
    REQUIRE(harmonic(0) == Approx(1.0));
}

/// verifies exponential_decay: v(k) = initial * rate^(k-1), never below the minimum
TEST_CASE("exponential_decay: initial * rate^(k-1), never below the minimum", "[schedules]")
{
    relearn::exponential_decay<> epsilon(1.0, 0.5, 0.1);
    REQUIRE(epsilon(1) == Approx(1.0));
    REQUIRE(epsilon(2) == Approx(0.5));
    REQUIRE(epsilon(3) == Approx(0.25));
    REQUIRE(epsilon(4) == Approx(0.125));
    // 0.0625 would be below the minimum
    REQUIRE(epsilon(5) == Approx(0.1));
    REQUIRE(epsilon(1000) == Approx(0.1));
}

/// verifies linear_decay: a straight line from initial to final over `steps` steps, then final
TEST_CASE("linear_decay: from initial to final over a number of steps", "[schedules]")
{
    relearn::linear_decay<> epsilon(1.0, 0.2, 4);
    REQUIRE(epsilon(1) == Approx(1.0));
    REQUIRE(epsilon(2) == Approx(0.8));
    REQUIRE(epsilon(3) == Approx(0.6));
    REQUIRE(epsilon(5) == Approx(0.2));
    REQUIRE(epsilon(100) == Approx(0.2));
    // no steps: final at once
    relearn::linear_decay<> instant(1.0, 0.2, 0);
    REQUIRE(instant(1) == Approx(0.2));
}

/// verifies that visit_counter counts each state/action pair on its own
TEST_CASE("visit_counter counts each pair on its own", "[schedules]")
{
    relearn::visit_counter<state,action> visits;
    REQUIRE(visits.count(state("s"), action("a")) == 0);
    REQUIRE(visits.increment(state("s"), action("a")) == 1);
    REQUIRE(visits.increment(state("s"), action("a")) == 2);
    REQUIRE(visits.increment(state("s"), action("b")) == 1);
    REQUIRE(visits.count(state("s"), action("a")) == 2);
    REQUIRE(visits.count(state("t"), action("a")) == 0);
    visits.clear();
    REQUIRE(visits.count(state("s"), action("a")) == 0);
}

/// verifies that α = 1/n per visit makes each value the plain average of its rewards,
/// for every learner with a learning rate (Sutton & Barto, Section 2.4)
TEST_CASE("learning_rate 1/n per visit gives the average reward", "[schedules]")
{
    // the rewards 1, 2, 3, 4 average 2.5; α = 1/n gives exactly that:
    // Q = 0 + 1 (1 - 0) = 1, then 1 + 1/2 (2 - 1) = 1.5, then 1.5 + 1/3 (3 - 1.5) = 2,
    // then 2 + 1/4 (4 - 2) = 2.5
    relearn::q_learning<state,action> q;
    REQUIRE(average_of_four(q) == Approx(2.5));
    REQUIRE(q.visits.count(state("s"), action("a")) == 4);

    relearn::sarsa<state,action> sarsa;
    REQUIRE(average_of_four(sarsa) == Approx(2.5));

    relearn::expected_sarsa<state,action> expected;
    REQUIRE(average_of_four(expected) == Approx(2.5));
}

/// verifies that each pair gets its own learning rate, from its own number of updates
TEST_CASE("learning_rate: each pair counts its own updates", "[schedules]")
{
    policy memory;
    relearn::q_learning<state,action> learner;
    learner.learning_rate = relearn::polynomial_decay<>(1.0, 1.0);
    // (s, a) is updated twice: 4, then 4 + 1/2 (8 - 4) = 6
    learner(one_step(4), memory);
    learner(one_step(8), memory);
    // (t, b) is updated once, so its first update uses α = 1: its value is its reward
    learner(episode {{state(10, "t"), action("b")}}, memory);
    REQUIRE(memory.value(state("s"), action("a")) == Approx(6));
    REQUIRE(memory.value(state("t"), action("b")) == Approx(10));
}

/// verifies that without a learning_rate, `alpha` is used and nothing is counted
TEST_CASE("learning_rate not set: alpha is used", "[schedules]")
{
    policy memory;
    relearn::sarsa<state,action> learner;
    learner.alpha = 0.5;
    // 0 + 0.5 (4 - 0) = 2, then 2 + 0.5 (4 - 2) = 3
    learner(one_step(4), memory);
    learner(one_step(4), memory);
    REQUIRE(memory.value(state("s"), action("a")) == Approx(3));
    REQUIRE(learner.visits.count(state("s"), action("a")) == 0);
}

/// verifies that brace initialisation, used by the examples, still works
TEST_CASE("q_learning can still be brace-initialised with alpha and gamma", "[schedules]")
{
    auto learner = relearn::q_learning<state,action>{0.25, 0.5};
    REQUIRE(learner.alpha == Approx(0.25));
    REQUIRE(learner.gamma == Approx(0.5));
    REQUIRE(!learner.learning_rate);
}
