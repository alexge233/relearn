#ifndef RELEARN_GAMES_RANDOM_WALK_HPP
#define RELEARN_GAMES_RANDOM_WALK_HPP
/**
 * The random walk, as in Sutton & Barto, Examples 6.2 (5 states) and 7.1 (19 states).
 *
 *     [end] 1 2 3 4 5 [end]          start in the middle
 *
 * - States 1 ... n; each step moves left or right with equal probability.
 *   There is a single action, "step": this is a *prediction* task, learning
 *   how much each state is worth, not what to do.
 * - Leaving on the right is worth `right_reward` (+1), on the left
 *   `left_reward` (0 in Example 6.2, -1 in Example 7.1); other steps are worth 0.
 * - With no discount the true value of state i is
 *   left_reward + (right_reward - left_reward) * i / (n + 1),
 *   e.g. 1/6, 2/6, ... 5/6 for Example 6.2.
 */
#include "game.hpp"
#include <random>
#include <vector>

namespace games
{
/**
 * @class random_walk
 * @brief a walk that moves left or right at random, until it leaves at either end
 */
class random_walk
{
public:
    /// a state is the position of the walk: 1 ... n
    using state_type  = unsigned int;
    /// there is only one action, 0: take a step
    using action_type = unsigned int;

    /**
     * @brief create a random walk
     * @param states is the number n of states (5 in Example 6.2, 19 in Example 7.1)
     * @param left_reward is the reward for leaving on the left
     * @param right_reward is the reward for leaving on the right
     * @param seed seeds the coin that decides each step
     */
    explicit random_walk(unsigned int states = 5,
                         double left_reward = 0,
                         double right_reward = 1,
                         unsigned int seed = 0)
    : n_(states),
      left_(left_reward),
      right_(right_reward),
      prng_(seed)
    {}

    /// @return the number of states n
    unsigned int states() const
    {
        return n_;
    }

    /// @return the true (undiscounted) value of state @param i
    double true_value(state_type i) const
    {
        // the chance of leaving on the right from state i is i / (n + 1)
        return left_ + (right_ - left_) * static_cast<double>(i) / static_cast<double>(n_ + 1);
    }

    /// @brief start a new episode @return the middle state
    state_type reset()
    {
        at_ = (n_ + 1) / 2;
        return at_;
    }

    /// @brief take a step left or right at random @return the state reached, the reward and whether the walk ended
    step_result<state_type> step(const action_type &)
    {
        // a fair coin decides the direction of the step
        std::bernoulli_distribution go_right(0.5);
        step_result<state_type> out;
        if (go_right(prng_)) {
            at_++;
        }
        else {
            at_--;
        }
        out.next = at_;
        // leaving on either end ends the walk, with that end's reward
        if (at_ == 0) {
            out.reward = left_;
            out.terminated = true;
        }
        else if (at_ == n_ + 1) {
            out.reward = right_;
            out.terminated = true;
        }
        return out;
    }

    /// @return the only action, 0
    std::vector<action_type> actions(const state_type &) const
    {
        return {0};
    }

private:
    // number of states
    unsigned int n_;
    // rewards for leaving on the left and on the right
    double left_;
    double right_;
    // the coin that decides each step
    std::mt19937 prng_;
    // the position of the walk
    unsigned int at_ = 0;
};
} // namespace games
#endif
