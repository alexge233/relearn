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
class bandit
{
public:
    using state_type  = unsigned int;    // there is only one state, 0
    using action_type = unsigned int;    // the arm, 0 ... arms-1

    explicit bandit(unsigned int arms = 10, unsigned int seed = 0)
    : prng_(seed)
    {
        std::normal_distribution<double> mean(0.0, 1.0);
        for (unsigned int a = 0; a < arms; a++) {
            means_.push_back(mean(prng_));
        }
    }

    /// the true mean reward q*(a) of arm @param a
    double mean(action_type a) const { return means_[a]; }
    /// the arm with the highest mean
    action_type best_arm() const
    {
        return static_cast<action_type>(std::max_element(means_.begin(), means_.end()) - means_.begin());
    }

    state_type reset() { return 0; }

    step_result<state_type> step(const action_type & a)
    {
        std::normal_distribution<double> reward(means_[a], 1.0);
        step_result<state_type> out;
        out.next = 0;
        out.reward = reward(prng_);
        out.terminated = true;
        return out;
    }

    std::vector<action_type> actions(const state_type &) const
    {
        std::vector<action_type> arms(means_.size());
        for (unsigned int a = 0; a < arms.size(); a++) arms[a] = a;
        return arms;
    }

private:
    std::mt19937 prng_;
    std::vector<double> means_;
};
} // namespace games
#endif
