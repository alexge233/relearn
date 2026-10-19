// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/game.hpp"
#include "../examples/games/cliff_walking.hpp"
#include <cmath>
#include <string>
#include "solvers.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of `sarsa` and `expected_sarsa` (Sutton & Barto, Sections 6.4 and 6.6):
 * - one update at a time, against values worked out by hand
 * - convergence to the exact values of the policy followed, q_π
 * - Cliff Walking (Example 6.6): the on-policy learners earn more while exploring
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;

/**
 * @brief a policy where the next state, "s1", has two actions of different value
 * @return Q(s1, b) = 10 (the best action) and Q(s1, c) = 2
 */
policy next_state_values()
{
    policy memory;
    memory.update(state("s1"), action("b"), 10);
    memory.update(state("s1"), action("c"), 2);
    return memory;
}

/**
 * @brief an episode cut short after one step: (s0, a) earned 1, and the agent then chose c in s1
 * @return the two links of the episode
 */
episode one_step_then_c()
{
    return episode {{state(1, "s0"), action("a")},
                    {state(0, "s1"), action("c")}};
}

/**
 * @class corridor
 * @brief a slippery corridor of 5 cells, 0 ... 4, with the goal at cell 4
 *
 * - actions: 0 moves left (staying put at cell 0), 1 moves right
 * - moving right slips with probability 0.2, and the agent stays put
 * - every move costs -1, and reaching cell 4 ends the episode
 */
class corridor
{
public:
    using state_type  = int;
    using action_type = unsigned int;

    /// @brief create the corridor @param seed seeds the slips
    explicit corridor(unsigned int seed)
    : prng_(seed)
    {}

    /// @brief start a new episode @return cell 0
    state_type reset()
    {
        at_ = 0;
        return at_;
    }

    /// @brief take the move @param a @return the cell reached, the reward and whether the game ended
    games::step_result<state_type> step(const action_type & a)
    {
        games::step_result<state_type> out;
        std::bernoulli_distribution slip(0.2);
        if (a == 1) {
            // moving right succeeds 80% of the time
            if (!slip(prng_)) {
                at_++;
            }
        }
        else if (at_ > 0) {
            // moving left always succeeds, except at the wall
            at_--;
        }
        out.reward = -1;
        out.terminated = (at_ == 4);
        out.next = at_;
        return out;
    }

    /// @return the legal moves: left and right, in every cell
    std::vector<action_type> actions(const state_type &) const
    {
        return {0, 1};
    }

    /**
     * @brief the exact model of the corridor, written from its rules
     * @param gamma is the discount rate
     * @return the model, for the solvers
     */
    static solvers::tabular model(double gamma)
    {
        solvers::tabular m;
        m.states  = 5;
        m.actions = 2;
        m.gamma   = gamma;
        m.model   = [](int s, int a) {
            std::vector<solvers::outcome> outcomes;
            // the goal ends the episode: no action there
            if (s == 4) {
                return outcomes;
            }
            if (a == 1) {
                outcomes.push_back({0.8, s + 1, -1, s + 1 == 4});
                outcomes.push_back({0.2, s, -1, false});
            }
            else {
                outcomes.push_back({1.0, s > 0 ? s - 1 : 0, -1, false});
            }
            return outcomes;
        };
        return m;
    }

private:
    // the cell the agent is in
    int at_ = 0;
    // the random slips
    std::mt19937 prng_;
};

using corridor_types = games::types<corridor>;

/**
 * @brief learn the corridor online, with an agent that moves at random (left or right, 50/50)
 * @param learner is the learner to train
 * @param seed seeds the corridor and the agent
 * @return the learnt policy
 *
 * The learning rate decays as α_k = 0.1 * 1000 / (1000 + k) for episode k,
 * so that the estimates settle down instead of staying noisy.
 */
template <class learner_type>
corridor_types::policy learn_corridor(learner_type & learner,
                                      unsigned int seed)
{
    corridor game(seed);
    std::mt19937 prng(seed + 100);
    corridor_types::policy memory;
    // the agent moves at random: its policy π does not depend on what is learnt
    auto agent = [&](const int &,
                     const std::vector<unsigned int> & legal) {
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        return legal[pick(prng)];
    };
    for (int k = 1; k <= 40000; k++) {
        learner.alpha = 0.1 * 1000.0 / (1000.0 + k);
        games::play_online(game, agent, learner, memory, 1000);
    }
    return memory;
}

/**
 * @brief the largest difference between the learnt values and exact ones
 * @param memory is the learnt policy
 * @param m is the exact model
 * @param v are the exact state values the action values are computed from
 * @return the largest |Q(s,a) - exact Q(s,a)| over the cells 0 ... 3 and both moves
 */
double worst_error(const corridor_types::policy & memory,
                   const solvers::tabular & m,
                   const std::vector<double> & v)
{
    double worst = 0;
    for (int s = 0; s < 4; s++) {
        for (unsigned int a = 0; a < 2; a++) {
            double learnt = memory.value(corridor_types::state(s), corridor_types::action(a));
            worst = std::max(worst, std::fabs(learnt - m.q(v, s, static_cast<int>(a))));
        }
    }
    return worst;
}

using cliff_types = games::types<games::cliff_walking>;

/**
 * @brief prepare a learner to learn Cliff Walking: nothing to do for most learners
 * @param learner is the learner
 * @param memory is the policy being learnt
 * @param epsilon is the probability of a random move
 */
template <class learner_type>
void prepare(learner_type &,
             const cliff_types::policy &,
             double)
{}

/**
 * @brief prepare Expected SARSA: give it the ε-greedy policy π the agent follows
 * @param learner is the learner
 * @param memory is the policy being learnt, which the agent is greedy on
 * @param epsilon is the probability of a random move
 */
void prepare(relearn::expected_sarsa<cliff_types::state,cliff_types::action> & learner,
             const cliff_types::policy & memory,
             double epsilon)
{
    games::cliff_walking rules;
    learner.probabilities = [&memory, rules, epsilon](const cliff_types::state & s) {
        return games::epsilon_greedy_probabilities<games::cliff_walking>(memory,
                                                                        s.trait(),
                                                                        rules.actions(s.trait()),
                                                                        epsilon);
    };
}

/**
 * @brief learn Cliff Walking online as in Sutton & Barto, Example 6.6: α = 0.5, ε = 0.1, γ = 1
 * @param learner is the learner to train
 * @param seed seeds the agent
 * @return the average return while exploring, over episodes 101 ... 500
 */
template <class learner_type>
double cliff_return(learner_type & learner,
                    unsigned int seed)
{
    const double epsilon = 0.1;
    games::cliff_walking game;
    std::mt19937 prng(seed);
    cliff_types::policy memory;
    learner.alpha = 0.5;
    learner.gamma = 1.0;
    prepare(learner, memory, epsilon);
    // the ε-greedy agent
    auto agent = [&](const games::position & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<games::cliff_walking>(memory, s, legal, epsilon, prng);
    };
    // play and learn 500 episodes, adding up the returns of the last 400
    double sum = 0;
    for (int i = 1; i <= 500; i++) {
        auto r = games::play_online(game, agent, learner, memory, 1000);
        if (i > 100) {
            sum += r.total_reward;
        }
    }
    return sum / 400;
}
} // namespace

/// verifies that SARSA learns from the action taken next, where Q-learning uses the best one
TEST_CASE("sarsa: learns from the action taken next, not the best one", "[sarsa]")
{
    // Q(s0, a) = 0 + α (r + γ Q(s1, c) - 0) = 1 + 0.5 * 2 = 2
    auto memory = next_state_values();
    relearn::sarsa<state,action> learner;
    learner.alpha = 1;
    learner.gamma = 0.5;
    learner(one_step_then_c(), memory, false);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(2));

    // Q-learning would use the best action of s1: 1 + 0.5 * 10 = 6
    auto other = next_state_values();
    relearn::q_learning<state,action> q;
    q.alpha = 1;
    q.gamma = 0.5;
    q(one_step_then_c(), other, false);
    REQUIRE(other.value(state("s0"), action("a")) == Approx(6));

    // the cut-short episode's last link is not learnt: s1's values are unchanged
    REQUIRE(memory.value(state("s1"), action("c")) == Approx(2));
}

/// verifies that the last link of a finished episode moves towards its reward only
TEST_CASE("sarsa: the last step of a finished episode learns its reward", "[sarsa]")
{
    // Q(s0, a) = 4 + α (r - 4) = 4 + 0.5 * (1 - 4) = 2.5
    policy memory;
    memory.update(state("s0"), action("a"), 4);
    relearn::sarsa<state,action> learner;
    learner.alpha = 0.5;
    learner(episode {{state(1, "s0"), action("a")}}, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(2.5));
}

/// verifies that Expected SARSA learns from the expected value of the next state under π
TEST_CASE("expected_sarsa: learns from the expected value of the next state", "[expected_sarsa]")
{
    // π takes b a quarter of the time and c the rest, so the next state is worth
    // 0.25 * 10 + 0.75 * 2 = 4, and Q(s0, a) = 1 + 0.5 * 4 = 3
    auto memory = next_state_values();
    relearn::expected_sarsa<state,action> learner;
    learner.alpha = 1;
    learner.gamma = 0.5;
    learner.probabilities = [](const state &) {
        return relearn::expected_sarsa<state,action>::distribution {{action("b"), 0.25},
                                                                    {action("c"), 0.75}};
    };
    REQUIRE(learner.expected_value(state("s1"), memory) == Approx(4));
    learner(one_step_then_c(), memory, false);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(3));
}

/// verifies that, without a policy π, Expected SARSA uses the greedy one, like Q-learning
TEST_CASE("expected_sarsa: without a policy it learns like Q-learning", "[expected_sarsa]")
{
    // greedy: the next state is worth its best value, 10, so Q(s0, a) = 1 + 0.5 * 10 = 6
    auto memory = next_state_values();
    relearn::expected_sarsa<state,action> learner;
    learner.alpha = 1;
    learner.gamma = 0.5;
    learner(one_step_then_c(), memory, false);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(6));
    // a state never seen is worth 0
    REQUIRE(learner.expected_value(state("unknown"), memory) == Approx(0));
}

/// verifies the probabilities of the ε-greedy choice, including ties
TEST_CASE("epsilon_greedy_probabilities: ε / |A| each, and the best actions share 1 - ε", "[sarsa][games]")
{
    using game_type = games::cliff_walking;
    using types     = games::types<game_type>;
    types::policy memory;
    games::position s {3, 1};
    // up and right are tied as best (never tried: worth 0), down and left are worse
    memory.update(types::state(s), types::action(games::down), -5);
    memory.update(types::state(s), types::action(games::left), -1);
    auto pi = games::epsilon_greedy_probabilities<game_type>(memory, s,
                                                             {games::up, games::right, games::down, games::left},
                                                             0.2);
    REQUIRE(pi.size() == 4);
    // 0.2 / 4 = 0.05 each, and the two best share 0.8
    REQUIRE(pi[0].second == Approx(0.45));
    REQUIRE(pi[1].second == Approx(0.45));
    REQUIRE(pi[2].second == Approx(0.05));
    REQUIRE(pi[3].second == Approx(0.05));
    double total = 0;
    for (const auto & item : pi) {
        total += item.second;
    }
    REQUIRE(total == Approx(1));
}

/// verifies that SARSA and Expected SARSA learn q_π, the values of the policy followed (Section 6.4),
/// while Q-learning, on the same kind of episodes, learns q*, the values of the best policy (Section 6.5)
TEST_CASE("sarsa and expected_sarsa converge to q_pi, q_learning to q*", "[sarsa][expected_sarsa][convergence]")
{
    // the exact values: of the random policy (π: left or right, 50/50), and of the best one
    const double gamma = 0.9;
    auto exact = corridor::model(gamma);
    std::vector<std::vector<double>> pi(5, {0.5, 0.5});
    auto v_pi   = exact.policy_values(pi);
    auto v_star = exact.optimal_values();
    // the two are far apart: from cell 0, moving right is worth -8.2 under π, but -4.1 at best
    REQUIRE(exact.q(v_pi, 0, 1) == Approx(-8.20812).epsilon(1e-4));
    REQUIRE(exact.q(v_star, 0, 1) == Approx(-4.05606).epsilon(1e-4));

    // tolerances: across seeds 1 ... 10, the worst errors measured were
    // 0.16 (SARSA), 0.13 (Expected SARSA) and 0.04 (Q-learning)
    SECTION("sarsa learns q_pi") {
        relearn::sarsa<corridor_types::state,corridor_types::action> learner;
        learner.gamma = gamma;
        auto memory = learn_corridor(learner, 1);
        REQUIRE(worst_error(memory, exact, v_pi) < 0.3);
    }
    SECTION("expected_sarsa, given π, learns q_pi") {
        relearn::expected_sarsa<corridor_types::state,corridor_types::action> learner;
        learner.gamma = gamma;
        learner.probabilities = [](const corridor_types::state &) {
            return relearn::expected_sarsa<corridor_types::state,corridor_types::action>::distribution
                {{corridor_types::action(0u), 0.5},
                 {corridor_types::action(1u), 0.5}};
        };
        auto memory = learn_corridor(learner, 2);
        REQUIRE(worst_error(memory, exact, v_pi) < 0.3);
    }
    SECTION("q_learning learns q_star from the same random play") {
        relearn::q_learning<corridor_types::state,corridor_types::action> learner;
        learner.gamma = gamma;
        auto memory = learn_corridor(learner, 3);
        REQUIRE(worst_error(memory, exact, v_star) < 0.1);
    }
}

/// verifies Sutton & Barto, Example 6.6 and Figure 6.3: while exploring with ε-greedy,
/// SARSA earns more than Q-learning (it learns to keep away from the cliff edge),
/// and Expected SARSA earns more than SARSA
TEST_CASE("cliff walking: expected_sarsa > sarsa > q_learning while exploring", "[sarsa][expected_sarsa][cliff]")
{
    // average over 10 runs, as the book averages over many runs
    double q = 0;
    double sarsa = 0;
    double expected = 0;
    for (unsigned int seed = 1; seed <= 10; seed++) {
        relearn::q_learning<cliff_types::state,cliff_types::action> q_learner;
        q += cliff_return(q_learner, seed) / 10;

        relearn::sarsa<cliff_types::state,cliff_types::action> sarsa_learner;
        sarsa += cliff_return(sarsa_learner, seed) / 10;

        relearn::expected_sarsa<cliff_types::state,cliff_types::action> expected_learner;
        expected += cliff_return(expected_learner, seed) / 10;
    }
    INFO("average return while exploring: q_learning " << q
         << ", sarsa " << sarsa << ", expected_sarsa " << expected);
    // across 6 groups of 10 seeds the averages were: q_learning -39 ... -43,
    // sarsa -27 ... -29.5, expected_sarsa -21 ... -22
    REQUIRE(sarsa > q + 5);
    REQUIRE(expected > sarsa + 3);
}
