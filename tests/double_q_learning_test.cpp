// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/maximization_bias.hpp"
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of `double_q_learning` (Sutton & Barto, Section 6.7):
 * - one update at a time, against values worked out by hand
 * - the coin that chooses which estimate to update
 * - the maximization bias of Example 6.7: Double Q-learning removes it
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;
using learner_type = relearn::double_q_learning<state,action>;

/**
 * @brief set the two estimates of the next state, "s1", so that they disagree:
 *        Q1 prefers b (10 against 2), Q2 prefers c (8 against 4)
 * @param learner is the learner whose estimates are set
 */
void disagree(learner_type & learner)
{
    learner.first.update(state("s1"), action("b"), 10);
    learner.first.update(state("s1"), action("c"), 2);
    learner.second.update(state("s1"), action("b"), 4);
    learner.second.update(state("s1"), action("c"), 8);
}

/**
 * @brief an episode cut short after one step: (s0, a) earned 1, and led to s1
 * @return the two links of the episode
 */
episode one_step_to_s1()
{
    return episode {{state(1, "s0"), action("a")},
                    {state(0, "s1"), action("b")}};
}

/**
 * @brief Q-learning has no coin: nothing to seed
 * @param learner is the learner
 * @param seed is not used
 */
template <class learner>
void seed_coin(learner &,
               unsigned int)
{}

/**
 * @brief seed Double Q-learning's coin, so that each run flips its own coins
 * @param learner is the learner
 * @param seed is the seed of the coin
 */
void seed_coin(relearn::double_q_learning<games::types<games::maximization_bias>::state,
                                          games::types<games::maximization_bias>::action> & learner,
               unsigned int seed)
{
    learner.coin.seed(seed);
}

/**
 * @brief the share of episodes going left from A, in episodes 201 ... 300 of Example 6.7
 * @param runs is the number of runs, each a new agent learning from scratch
 * @return the share, averaged over the runs
 *
 * The settings of Sutton & Barto, Example 6.7: ε-greedy with ε = 0.1, α = 0.1, γ = 1,
 * and 10 actions in B.
 */
template <class learner>
double late_share_left(unsigned int runs)
{
    using game_type = games::maximization_bias;
    using types     = games::types<game_type>;
    double left = 0;
    for (unsigned int run = 0; run < runs; run++) {
        game_type game(10, run);
        std::mt19937 prng(run + 5000);
        types::policy memory;
        learner agent_learner;
        agent_learner.alpha = 0.1;
        agent_learner.gamma = 1.0;
        seed_coin(agent_learner, run + 9000);
        // the ε-greedy agent
        auto agent = [&](const unsigned int & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        for (unsigned int i = 0; i < 300; i++) {
            auto r = games::play(game, agent, 10);
            agent_learner(r.episode, memory, r.terminated);
            // an episode of two steps went left (through B)
            if (i >= 200) {
                left += (r.steps == 2);
            }
        }
    }
    return left / (runs * 100.0);
}
} // namespace

/// verifies that each update picks the next action with one estimate and values it with the other
TEST_CASE("double_q_learning: one estimate picks the next action, the other values it", "[double_q]")
{
    // with α = 1 and γ = 0.5, from r = 1:
    // - updating Q1: Q1 picks b in s1, and Q2 values it at 4, so Q1(s0, a) = 1 + 0.5 * 4 = 3
    // - updating Q2: Q2 picks c in s1, and Q1 values it at 2, so Q2(s0, a) = 1 + 0.5 * 2 = 2
    // Q-learning would use the best value of s1, 10: 1 + 0.5 * 10 = 6
    bool updated_first  = false;
    bool updated_second = false;
    for (unsigned int seed = 0; seed < 20; seed++) {
        learner_type learner(1.0, 0.5, seed);
        disagree(learner);
        policy memory;
        learner(one_step_to_s1(), memory, false);
        double q1 = learner.first.value(state("s0"), action("a"));
        double q2 = learner.second.value(state("s0"), action("a"));
        // exactly one estimate is updated
        if (q1 != 0) {
            REQUIRE(q1 == Approx(3));
            REQUIRE(q2 == Approx(0));
            updated_first = true;
        }
        else {
            REQUIRE(q2 == Approx(2));
            updated_second = true;
        }
        // the agent acts on the average of both estimates
        REQUIRE(memory.value(state("s0"), action("a")) == Approx((q1 + q2) / 2));
    }
    // across 20 seeds, the coin chose each estimate at least once
    REQUIRE(updated_first);
    REQUIRE(updated_second);
}

/// verifies that the last step of a finished episode moves towards its reward only
TEST_CASE("double_q_learning: the last step of a finished episode learns its reward", "[double_q]")
{
    // whichever estimate is updated: 0 + 0.5 * (4 - 0) = 2, and the average is 1
    learner_type learner(0.5, 0.9, 3);
    policy memory;
    learner(episode {{state(4, "s0"), action("a")}}, memory);
    double q1 = learner.first.value(state("s0"), action("a"));
    double q2 = learner.second.value(state("s0"), action("a"));
    REQUIRE(q1 + q2 == Approx(2));
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(1));
}

/// verifies that the coin updates each estimate about half the time
TEST_CASE("double_q_learning: the coin updates each estimate about half the time", "[double_q]")
{
    // 1000 different states, each updated once: count the ones that went to Q1
    learner_type learner(1.0, 0.9, 7);
    policy memory;
    int first = 0;
    for (int i = 0; i < 1000; i++) {
        const std::string name = "s" + std::to_string(i);
        learner(episode {{state(1, name), action("a")}}, memory);
        if (learner.first.value(state(name), action("a")) != 0) {
            first++;
        }
    }
    // a fair coin: 500 on average, with a standard deviation of about 16
    REQUIRE(first > 430);
    REQUIRE(first < 570);
}

/// verifies that with α = 1/n, each estimate counts its own updates, and is the average of its targets
TEST_CASE("double_q_learning: each estimate counts its own updates", "[double_q][schedules]")
{
    learner_type learner(0.9, 0.9, 11);
    learner.learning_rate = relearn::polynomial_decay<>(1.0, 1.0);
    policy memory;
    // the same reward every time: the first update of each estimate (α = 1) already learns it
    for (int i = 0; i < 20; i++) {
        learner(episode {{state(4, "s0"), action("a")}}, memory);
    }
    auto in_first  = learner.first_visits.count(state("s0"), action("a"));
    auto in_second = learner.second_visits.count(state("s0"), action("a"));
    REQUIRE(in_first + in_second == 20u);
    REQUIRE(in_first > 0u);
    REQUIRE(in_second > 0u);
    REQUIRE(learner.first.value(state("s0"), action("a")) == Approx(4));
    REQUIRE(learner.second.value(state("s0"), action("a")) == Approx(4));
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(4));
}

/// verifies Sutton & Barto, Example 6.7 and Figure 6.5: Q-learning's maximization bias makes it
/// go left far more often than the best ε-greedy agent (5%), and Double Q-learning removes it
TEST_CASE("maximization bias: double_q_learning goes left about as rarely as it should", "[double_q][max_bias]")
{
    // across 6 groups of 1000 runs, in episodes 201 ... 300 the share going left was
    // 6.4% ... 7.3% for Double Q-learning, and 18.4% ... 20.5% for Q-learning
    double q       = late_share_left<relearn::q_learning<games::types<games::maximization_bias>::state,
                                                         games::types<games::maximization_bias>::action>>(1000);
    double double_q = late_share_left<relearn::double_q_learning<games::types<games::maximization_bias>::state,
                                                                 games::types<games::maximization_bias>::action>>(1000);
    INFO("share going left in episodes 201-300: q_learning " << q << ", double_q_learning " << double_q);
    REQUIRE(double_q < 0.09);
    REQUIRE(q > 0.15);
}
