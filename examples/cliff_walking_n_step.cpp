/**
 * Cliff Walking (Sutton & Barto, Example 6.6) learnt with the n-step methods of Chapter 7, with n = 4.
 *
 * 1. On-policy: the agent plays ε-greedy (ε = 0.1) and learns the values of that same policy, with
 *    n-step SARSA, n-step Expected SARSA, n-step Q(σ) (σ = 0.5) and n-step tree backup. They all learn
 *    a safe path away from the edge, as one-step SARSA does; the more a learner uses expectations
 *    instead of the actions taken (tree backup the most), the more it earns whilst exploring.
 * 2. Off-policy: the agent still plays ε-greedy, but learns the values of the *greedy* policy, q*,
 *    whose path runs right along the edge: with off-policy n-step SARSA and n-step Expected SARSA
 *    (importance sampling), Q(σ) (importance sampling on half of each step) and tree backup (none).
 *    The less importance sampling, the closer to q* they get in the same number of episodes. And
 *    one-step Q-Learning, for reference, gets there first: every move costs -1 here, so looking
 *    4 rewards ahead brings little, whereas cutting the return at every exploring move costs a lot.
 *
 * Each learner is trained in 20 runs of 1000 episodes, learning after every step, with α = 0.2
 * and γ = 1. The program succeeds if, on-policy, Q(σ) and tree backup earn more whilst exploring
 * than n-step SARSA, and if, off-policy, the errors against q* come in the order Q-Learning,
 * tree backup, Q(σ), off-policy n-step SARSA, as the least importance sampling learns best.
 *
 * @see games/cliff_walking.hpp for the rules
 * @see docs/n_step.md for the algorithms
 */
#include "games/cliff_walking.hpp"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// set shortcuts to the game and to its relearn types
using game_type = games::cliff_walking;
using types     = games::types<game_type>;

// the ε of Sutton & Barto, Example 6.6, a smaller α, and n = 4
const double epsilon = 0.1;
const double alpha   = 0.2;
const unsigned int n        = 4;
const unsigned int runs     = 20;
const unsigned int episodes = 1000;

/// what one learner did, over all its runs
struct outcome
{
    /// the average return whilst exploring, over episodes 101 ... 1000 of every run
    double exploring = 0;
    /// the number of runs whose greedy policy learnt the path along the edge (13 moves)
    unsigned int edge = 0;
    /// the number of runs whose greedy policy reached the goal at all, in 100 moves
    unsigned int reached = 0;
    /// the error of the values learnt along the shortest path, against the optimal values, averaged over the runs
    double error = 0;
};

/**
 * @brief how far the learnt values of the shortest path are from the optimal ones, q*
 * @param memory is the learnt policy
 * @return the average |Q(s,a) - q*(s,a)| over the 13 moves of the path along the edge:
 *         up from the start, q* = -13; right along the edge from cell x, q* = x - 12; down to the goal, q* = -1
 */
double path_error(const types::policy & memory)
{
    double error = std::fabs(memory.value(types::state(game_type::start()), types::action(0)) + 13);
    for (int x = 0; x < game_type::width - 1; x++) {
        error += std::fabs(memory.value(types::state(games::position{x, game_type::height - 2}), types::action(1)) - (x - 12));
    }
    error += std::fabs(memory.value(types::state(games::position{game_type::width - 1, game_type::height - 2}), types::action(2)) + 1);
    return error / 13;
}

/**
 * @brief the ε-greedy policy the agent plays, as each move with its probability
 * @param memory is the policy being learnt, which the agent is greedy on
 * @param s is the state
 * @param e is ε: 0.1 for the agent, 0 for the greedy policy being learnt off-policy
 * @return π(· | s)
 */
std::vector<std::pair<types::action,double>> epsilon_greedy(const types::policy & memory,
                                                            const types::state & s,
                                                            double e)
{
    game_type rules;
    return games::epsilon_greedy_probabilities<game_type>(memory, s.trait(), rules.actions(s.trait()), e);
}

/**
 * @brief follow the greedy policy from the start, for up to 100 moves
 * @param memory is the learnt policy
 * @param moves gets the number of moves, if it reached the goal
 * @return true if it reached the goal
 */
bool greedy_path(const types::policy & memory,
                 unsigned int & moves)
{
    game_type game;
    std::mt19937 prng(0);
    auto s = game.reset();
    for (moves = 1; moves <= 100; moves++) {
        auto out = game.step(games::greedy<game_type>(memory, s, game.actions(s), prng));
        if (out.terminated) {
            return true;
        }
        s = out.next;
    }
    return false;
}

/**
 * @brief train a learner in `runs` runs of `episodes` episodes each, the agent playing ε-greedy
 * @param learner is the learner, with its α, γ and n
 * @param prepare gives the learner, once its policy exists, π (and b): `prepare(learner, memory)`
 * @return the average return whilst exploring, and the greedy paths learnt
 */
template <class learner_type,
          class preparer>
outcome train(learner_type learner,
              preparer && prepare)
{
    outcome result;
    for (unsigned int run = 1; run <= runs; run++) {
        // a new agent for each run: an empty policy and its own random numbers
        game_type game;
        std::mt19937 prng(run);
        types::policy memory;
        learner_type fresh = learner;
        prepare(fresh, memory);
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, epsilon, prng);
        };
        // play and learn after every step, the last n steps at a time
        for (unsigned int i = 0; i < episodes; i++) {
            auto r = games::play_online(game, agent, fresh, memory, 1000, fresh.n);
            if (i >= 100) {
                result.exploring += r.total_reward / ((episodes - 100) * runs);
            }
        }
        // the greedy path learnt in this run: 13 moves is the one along the edge
        result.error += path_error(memory) / runs;
        unsigned int moves = 0;
        if (greedy_path(memory, moves)) {
            result.reached++;
            result.edge += (moves == 13);
        }
    }
    return result;
}

/**
 * @brief print one learner's outcome
 * @param name is the learner's name
 * @param o is what it did
 * @param with_error is true if the learner learns q*, so that its error against q* means something
 */
void print(const std::string & name,
           const outcome & o,
           bool with_error)
{
    // std::setw counts bytes: widen it by the extra bytes of the letters such as σ, so that the columns line up
    int extra = 0;
    for (unsigned char c : name) {
        extra += (c & 0xC0) == 0x80;
    }
    std::cout << "    " << std::left << std::setw(32 + extra) << name << std::right
              << std::setw(9) << o.exploring
              << std::setw(8) << o.reached << "/" << runs
              << std::setw(9) << o.edge << "/" << runs;
    if (with_error) {
        std::cout << std::setw(9) << o.error;
    }
    std::cout << std::endl;
}

int main()
{
    using sarsa_type    = relearn::n_step_sarsa<types::state, types::action>;
    using expected_type = relearn::n_step_expected_sarsa<types::state, types::action>;
    using sigma_type    = relearn::n_step_q_sigma<types::state, types::action>;
    using tree_type     = relearn::n_step_tree_backup<types::state, types::action>;
    using off_type      = relearn::off_policy_n_step_sarsa<types::state, types::action>;

    // π for the on-policy learners: the ε-greedy policy the agent plays
    auto on_policy = [](auto & learner, const types::policy & memory) {
        learner.probabilities = [&memory](const types::state & s) {
            return epsilon_greedy(memory, s, epsilon);
        };
    };
    // π for the off-policy learners: greedy; and b, the ε-greedy policy the agent plays
    auto greedy = [](auto & learner, const types::policy & memory) {
        learner.probabilities = [&memory](const types::state & s) {
            return epsilon_greedy(memory, s, 0.0);
        };
    };
    auto greedy_from_b = [&greedy](auto & learner, const types::policy & memory) {
        greedy(learner, memory);
        learner.behaviour = [&memory](const types::state & s, const types::action & a) {
            for (const auto & item : epsilon_greedy(memory, s, epsilon)) {
                if (item.first == a) {
                    return item.second;
                }
            }
            return 0.0;
        };
    };
    auto nothing = [](auto &, const types::policy &) {};

    std::cout << "Cliff Walking, n = " << n << ", α = " << alpha << ", ε = " << epsilon
              << ", " << runs << " runs of " << episodes << " episodes" << std::endl << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "    learner                         exploring   reached   the edge    error" << std::endl;

    // 1. on-policy: the values of the ε-greedy policy the agent plays
    std::cout << "  on-policy (π is the ε-greedy policy played):" << std::endl;
    std::vector<outcome> on;
    on.push_back(train(sarsa_type(alpha, 1.0, n), nothing));
    print("n-step SARSA", on.back(), false);
    on.push_back(train(expected_type(alpha, 1.0, n), on_policy));
    print("n-step Expected SARSA", on.back(), false);
    on.push_back(train(sigma_type(alpha, 1.0, n, 0.5), on_policy));
    print("n-step Q(σ), σ = 0.5", on.back(), false);
    on.push_back(train(tree_type(alpha, 1.0, n), on_policy));
    print("n-step tree backup", on.back(), false);

    // 2. off-policy: the values of the greedy policy, from the ε-greedy play
    std::cout << "  off-policy (π is greedy, b the ε-greedy policy played):" << std::endl;
    std::vector<outcome> off;
    auto q_learning = train(tree_type(alpha, 1.0, 1), greedy);
    print("tree backup, n = 1 (Q-Learning)", q_learning, true);
    off.push_back(train(off_type(alpha, 1.0, n), greedy_from_b));
    print("off-policy n-step SARSA", off.back(), true);
    off.push_back(train(expected_type(alpha, 1.0, n), greedy_from_b));
    print("n-step Expected SARSA", off.back(), true);
    off.push_back(train(sigma_type(alpha, 1.0, n, 0.5), greedy_from_b));
    print("n-step Q(σ), σ = 0.5", off.back(), true);
    off.push_back(train(tree_type(alpha, 1.0, n), greedy));
    print("n-step tree backup", off.back(), true);

    std::cout << std::endl
              << "  exploring: the average return whilst exploring, episodes 101 to " << episodes << std::endl
              << "  reached:   the runs whose greedy policy reaches the goal" << std::endl
              << "  the edge:  the runs whose greedy policy takes the shortest path, along the edge (13 moves)" << std::endl
              << "  error:     the average |Q - q*| of the 13 moves of that path, for the learners of q*"
              << std::endl;

    // on-policy, the expectations earn more whilst exploring: Q(σ) and tree backup more than n-step SARSA;
    // off-policy, the less importance sampling, the closer to q*: Q-Learning, tree backup, Q(σ), n-step SARSA
    bool as_expected = (on[2].exploring > on[0].exploring) &&
                       (on[3].exploring > on[0].exploring) &&
                       (q_learning.error < off[3].error) &&
                       (off[3].error < off[2].error) &&
                       (off[2].error < off[0].error);
    return as_expected ? EXIT_SUCCESS : EXIT_FAILURE;
}
