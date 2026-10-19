#ifndef RELEARN_GAMES_BANDIT_HPP
#define RELEARN_GAMES_BANDIT_HPP
/**
 * The k-armed bandit testbed, as in Sutton & Barto, Section 2.3.
 *
 * - One state; k actions (arms). Each episode is a single pull.
 * - Arm a pays a reward drawn from a normal distribution with mean q*(a) and
 *   variance 1. The means q*(a) are themselves drawn from a standard normal
 *   distribution when the bandit is created.
 * - The task is the trade-off between exploring arms and exploiting the best
 *   one found so far.
 */
#include "game.hpp"
#include <algorithm>
#include <random>
#include <vector>

namespace games
{
/**
 * @class bandit
 * @brief the k-armed bandit: pull one of k arms, each paying a random reward around its own mean
 */
class bandit
{
public:
    /// there is only one state, 0
    using state_type  = unsigned int;
    /// an action is the arm pulled, 0 ... arms-1
    using action_type = unsigned int;

    /**
     * @brief a new bandit
     * @param arms is the number of arms
     * @param seed seeds the generator of the means and of the rewards
     */
    explicit bandit(unsigned int arms = 10,
                    unsigned int seed = 0)
    : prng_(seed)
    {
        // the true mean of each arm, drawn from a standard normal distribution
        std::normal_distribution<double> mean(0.0, 1.0);
        for (unsigned int a = 0; a < arms; a++) {
            means_.push_back(mean(prng_));
        }
    }

    /// @return the true mean reward q*(a) of arm @param a
    double mean(action_type a) const
    {
        return means_[a];
    }

    /// @return the arm with the highest mean
    action_type best_arm() const
    {
        auto best = std::max_element(means_.begin(), means_.end());
        return static_cast<action_type>(best - means_.begin());
    }

    /// @brief start a new episode @return the only state, 0
    state_type reset()
    {
        return 0;
    }

    /// @brief pull arm @param a @return the reward; every pull ends the episode
    step_result<state_type> step(const action_type & a)
    {
        // the reward is drawn around the arm's mean, with variance 1
        std::normal_distribution<double> reward(means_[a], 1.0);
        step_result<state_type> out;
        out.next       = 0;
        out.reward     = reward(prng_);
        out.terminated = true;
        return out;
    }

    /// @return the legal actions: every arm
    std::vector<action_type> actions(const state_type &) const
    {
        std::vector<action_type> arms(means_.size());
        for (unsigned int a = 0; a < arms.size(); a++) {
            arms[a] = a;
        }
        return arms;
    }

private:
    // the generator of the means and of the rewards
    std::mt19937 prng_;
    // the true mean of each arm
    std::vector<double> means_;
};
} // namespace games
#endif
