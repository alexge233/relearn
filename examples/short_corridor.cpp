/**
 * The short corridor with switched actions (Sutton & Barto, Example 13.1), learnt with
 * the policy gradient methods of Chapter 13, which learn the policy itself:
 *
 * 1. REINFORCE (Section 13.3, Figure 13.1), with α = 2^-13
 * 2. REINFORCE with a baseline (Section 13.4, Figure 13.2), with α^θ = 2^-9 and α^w = 2^-6
 * 3. one-step Actor-Critic (Section 13.5), with the same step sizes as 2.
 *
 * The agent cannot tell the three cells apart, so all it can learn is how often to go right,
 * and the best policy is stochastic: right with probability 2 - √2 = 0.5858, worth -11.657.
 * Each method learns in 100 runs of 1000 episodes, all starting from π(right) = 0.05 (the
 * ε-greedy-left policy of Example 13.1, worth -82.1), γ = 1. The program prints the average
 * total reward per episode, as in Figures 13.1 and 13.2, and the policy learnt.
 *
 * Actor-Critic's critic cannot tell the cells apart either, and it misleads the actor: it learns
 * to go right almost always. So the program then shows the agent which cell it is in, and
 * Actor-Critic (α^θ = 2^-4, α^w = 2^-3) learns the best policy there, right, left, right, worth -3.
 *
 * The program succeeds if, in the median run, both REINFORCEs learn a policy worth more than -12.7,
 * REINFORCE with a baseline earns more than REINFORCE in episodes 101 to 200, Actor-Critic learns
 * p > 0.99 when the cells look the same, and a policy worth more than -3.1 when they are shown.
 *
 * @see games/short_corridor.hpp for the rules
 */
#include "games/short_corridor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

// set shortcuts to the game and to its relearn types
using game_type = games::short_corridor;
using types     = games::types<game_type>;
using reinforce_type = relearn::reinforce<types::state, types::action>;
using baseline_type  = relearn::reinforce_with_baseline<types::state, types::action>;
using critic_type    = relearn::actor_critic<types::state, types::action>;

/// @return the legal actions, for the learners: left and right, in every cell
std::vector<types::action> left_right(const types::state &)
{
    return {types::action(game_type::go_left), types::action(game_type::go_right)};
}

/// what was learnt in 100 runs
struct learnt
{
    /// the total reward of each episode, averaged over the runs
    std::vector<double> rewards;
    /// π(right) learnt in each run, in each cell
    std::vector<std::vector<double>> right;
    /// the exact value of the start, v(S), of the policy learnt in each run
    std::vector<double> values;
};

/**
 * @brief REINFORCE, with or without a baseline: play a whole episode, then learn from it
 * @return the played episode
 */
template <class learner_type,
          class chooser>
games::played<game_type> learn_episode(game_type & game,
                                       chooser && agent,
                                       learner_type & learner,
                                       types::policy & preferences)
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
games::played<game_type> learn_episode(game_type & game,
                                       chooser && agent,
                                       critic_type & learner,
                                       types::policy & preferences)
{
    return games::play_online_actor_critic(game, agent, learner, preferences, 1000);
}

/**
 * @brief learn the corridor in 100 runs of 1000 episodes, each cut short after 1000 steps
 * @param fresh is the learner each run starts from
 * @param shown is true for the agent to see which cell it is in
 * @return the rewards, and the policy learnt in each run
 */
template <class learner_type>
learnt learn(const learner_type & fresh,
             bool shown)
{
    const unsigned int runs = 100;
    const unsigned int episodes = 1000;
    const types::action right(game_type::go_right);
    learnt result;
    result.rewards.assign(episodes, 0);
    for (unsigned int run = 0; run < runs; run++) {
        std::mt19937 prng(run);
        game_type game;
        game.show_position(shown);
        learner_type learner = fresh;
        types::policy preferences;
        // when the cells look the same, start from π(right) = 0.05: h(right) - h(left) = ln(0.05 / 0.95)
        if (!shown) {
            const double half = std::log(19.0) / 2;
            preferences.update(types::state(0), right, -half);
            preferences.update(types::state(0), types::action(game_type::go_left), half);
        }
        // the agent draws each action from π, the policy being learnt
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<game_type::action_type> &) {
            return learner.sample(types::state(s), preferences, prng).trait();
        };
        for (unsigned int e = 0; e < episodes; e++) {
            auto r = learn_episode(game, agent, learner, preferences);
            result.rewards[e] += r.total_reward / runs;
        }
        // π(right) in each cell, and the exact value of the policy learnt
        std::vector<double> p;
        for (unsigned int cell = 0; cell < 3; cell++) {
            p.push_back(learner.probability(types::state(shown ? cell : 0), right, preferences));
        }
        result.right.push_back(p);
        result.values.push_back(shown ? game_type::start_value(p[0], p[1], p[2])
                                      : game_type::start_value(p[0]));
    }
    return result;
}

/// @return the median of @param values (the upper one of the two middle values)
double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

/**
 * @return the median run of @param runs: the run whose policy is worth the median value
 *         (the upper one of the two middle values), so that its π(right) and its value
 *         are printed together, as one policy
 */
std::size_t median_run(const learnt & runs)
{
    std::vector<std::size_t> order(runs.values.size());
    for (std::size_t i = 0; i < order.size(); i++) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&runs](std::size_t a, std::size_t b) {
        return runs.values[a] < runs.values[b];
    });
    return order[order.size() / 2];
}

/// @return the mean reward of episodes @param first to @param last (counted from 1)
double mean_reward(const learnt & runs,
                   unsigned int first,
                   unsigned int last)
{
    double sum = 0;
    for (unsigned int e = first; e <= last; e++) {
        sum += runs.rewards[e - 1];
    }
    return sum / (last - first + 1);
}

int main()
{
    std::cout << "The short corridor with switched actions (Example 13.1):" << std::endl
              << "    [S] [x] [ ] [G]    in x, left and right are switched, and the cells look the same" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "    exact values: π(right) = 0.05 is worth " << game_type::start_value(0.05)
              << ", the best, π(right) = " << game_type::best_right()
              << ", is worth " << game_type::start_value(game_type::best_right()) << std::endl;

    // 1. the three methods, on the corridor whose cells look the same
    const double alpha_theta = std::pow(2.0, -9);
    const double alpha_w     = std::pow(2.0, -6);
    auto plain    = learn(reinforce_type(std::pow(2.0, -13), 1.0, left_right), false);
    auto baseline = learn(baseline_type(alpha_theta, alpha_w, 1.0, left_right), false);
    auto critic   = learn(critic_type(alpha_theta, alpha_w, 1.0, left_right), false);

    std::cout << std::endl << "average total reward per episode, over 100 runs (Figures 13.1 and 13.2):" << std::endl;
    std::cout << "    episodes      REINFORCE   with baseline   Actor-Critic" << std::endl;
    const std::vector<std::pair<unsigned int, unsigned int>> blocks = {{1, 10}, {11, 50}, {51, 100},
                                                                       {101, 200}, {201, 500}, {501, 1000}};
    std::cout << std::setprecision(1);
    for (const auto & block : blocks) {
        std::cout << "    " << std::setw(4) << block.first << " to " << std::setw(4) << block.second
                  << std::setw(11) << mean_reward(plain, block.first, block.second)
                  << std::setw(16) << mean_reward(baseline, block.first, block.second)
                  << std::setw(15) << mean_reward(critic, block.first, block.second) << std::endl;
    }

    std::cout << std::endl << "the policy learnt after 1000 episodes, in the median run:" << std::endl;
    std::cout << std::setprecision(3);
    const std::vector<std::pair<std::string, const learnt *>> methods = {
        {"REINFORCE              ", &plain},
        {"REINFORCE with baseline", &baseline},
        {"Actor-Critic           ", &critic}
    };
    for (const auto & method : methods) {
        const auto run = median_run(*method.second);
        std::cout << "    " << method.first << "  π(right) = " << std::setw(5) << method.second->right[run][0]
                  << ", worth " << std::setw(9) << method.second->values[run] << std::endl;
    }

    // 2. Actor-Critic again, with the cells shown
    auto shown = learn(critic_type(std::pow(2.0, -4), std::pow(2.0, -3), 1.0, left_right), true);
    const auto middle = median_run(shown);
    std::cout << std::endl << "Actor-Critic with the cells shown, after 1000 episodes, in the median run:" << std::endl
              << "    π(right) = " << shown.right[middle][0] << ", " << shown.right[middle][1]
              << ", " << shown.right[middle][2] << " in the three cells, worth " << shown.values[middle]
              << " (the best, right, left, right, is worth -3)" << std::endl;

    bool good = median(plain.values) > -12.7 &&
                median(baseline.values) > -12.7 &&
                mean_reward(baseline, 101, 200) > mean_reward(plain, 101, 200) &&
                critic.right[median_run(critic)][0] > 0.99 &&
                median(shown.values) > -3.1;
    return good ? EXIT_SUCCESS : EXIT_FAILURE;
}
