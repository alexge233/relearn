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
class random_walk
{
public:
    using state_type  = unsigned int;    // 1 ... n
    using action_type = unsigned int;    // the only action, 0

    explicit random_walk(unsigned int states = 5,
                         double left_reward = 0,
                         double right_reward = 1,
                         unsigned int seed = 0)
    : n_(states), left_(left_reward), right_(right_reward), prng_(seed)
    {}

    unsigned int states() const { return n_; }
    /// the true (undiscounted) value of state @param i
    double true_value(state_type i) const
    {
        return left_ + (right_ - left_) * static_cast<double>(i) / static_cast<double>(n_ + 1);
    }

    state_type reset()
    {
        at_ = (n_ + 1) / 2;
        return at_;
    }

    step_result<state_type> step(const action_type &)
    {
        std::bernoulli_distribution go_right(0.5);
        step_result<state_type> out;
        if (go_right(prng_)) {
            at_++;
        }
        else {
            at_--;
        }
        out.next = at_;
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

    std::vector<action_type> actions(const state_type &) const
    {
        return {0};
    }

private:
    unsigned int n_;
    double left_;
    double right_;
    std::mt19937 prng_;
    unsigned int at_ = 0;
};
} // namespace games
#endif
