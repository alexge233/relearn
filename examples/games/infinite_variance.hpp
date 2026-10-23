#ifndef RELEARN_GAMES_INFINITE_VARIANCE_HPP
#define RELEARN_GAMES_INFINITE_VARIANCE_HPP
/**
 * The one-state task of Sutton & Barto, Example 5.5, where ordinary importance sampling has an infinite variance.
 *
 *          left: back to s with 0.9, reward 0
 *     +---------+
 *     v         |
 *    [s] -------+--- left: the end with 0.1, reward +1
 *     |
 *     +--- right: the end, reward 0
 *
 * - A single state, s, and two actions: right ends the episode at once, worth 0; left goes
 *   back to s with probability 0.9, worth 0, and ends the episode with probability 0.1, worth +1.
 * - The target policy always goes left, so every one of its episodes earns +1: v_π(s) = 1 (with γ = 1).
 * - The behaviour policy goes left and right with probability 1/2 each. An episode that went right
 *   has a ratio of 0; one that went left k times and then ended has a ratio of 2^k, and the larger k,
 *   the rarer it is: the ratios times the returns have a mean of 1, but their variance is infinite.
 */
#include "game.hpp"
#include <random>
#include <vector>

namespace games
{
/**
 * @class infinite_variance
 * @brief one state, and a choice between ending the episode now (right) or, most of the time, coming back (left)
 */
class infinite_variance
{
public:
    /// there is a single state, 0
    using state_type  = unsigned int;
    /// an action is `left` or `right`
    using action_type = unsigned int;
    /// the two actions
    enum : action_type
    {
        left  = 0,
        right = 1
    };

    /// @brief create the task @param seed seeds whether left comes back
    explicit infinite_variance(unsigned int seed = 0)
    : prng_(seed)
    {}

    /// @brief start a new episode @return the only state, 0
    state_type reset()
    {
        return 0;
    }

    /// @brief take the action @param a @return the state reached, the reward and whether the episode ended
    step_result<state_type> step(const action_type & a)
    {
        step_result<state_type> out;
        out.next = 0;
        if (a == right) {
            // right: the end, worth 0
            out.terminated = true;
            return out;
        }
        // left: back to s with 0.9, or the end with 0.1, worth +1
        std::bernoulli_distribution ends(0.1);
        if (ends(prng_)) {
            out.reward = 1;
            out.terminated = true;
        }
        return out;
    }

    /// @return the legal actions: left and right
    std::vector<action_type> actions(const state_type &) const
    {
        return {left, right};
    }

private:
    // decides whether left comes back to s
    std::mt19937 prng_;
};
} // namespace games
#endif
