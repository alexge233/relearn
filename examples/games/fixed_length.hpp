#ifndef RELEARN_GAMES_FIXED_LENGTH_HPP
#define RELEARN_GAMES_FIXED_LENGTH_HPP
/**
 * Episodes that are always the same length, with a reward at every step: the case of Sutton & Barto's
 * Sections 5.8 and 5.9, where ordinary importance sampling scales every return by the ratios of every
 * action of the episode, even those that came after the rewards it is made of.
 *
 *     0 --> 1 --> 2 --> ... --> n-1 --> the end      whatever the action
 *
 * - The state is the number of steps taken so far, 0 ... n - 1, and every episode is n steps long.
 * - Action 0 earns 1, action 1 earns 0.
 * - A policy that takes action 0 with probability p is worth p (1 - γ^(n-t)) / (1 - γ) from state t:
 *   `value(t, p, gamma)`.
 */
#include "game.hpp"
#include <cmath>
#include <vector>

namespace games
{
/**
 * @class fixed_length
 * @brief n steps, whatever the agent does, and a reward of 1 for every action 0
 */
class fixed_length
{
public:
    /// a state is the number of steps taken: 0 ... n - 1
    using state_type  = unsigned int;
    /// an action is 0 (earns 1) or 1 (earns 0)
    using action_type = unsigned int;

    /// @brief create the task, whose episodes are @param steps long
    explicit fixed_length(unsigned int steps = 20)
    : n_(steps)
    {}

    /// @return the number of steps of every episode
    unsigned int steps() const
    {
        return n_;
    }

    /// @brief start a new episode @return state 0
    state_type reset()
    {
        at_ = 0;
        return at_;
    }

    /// @brief take the action @param a @return the next state, the reward (1 for action 0) and whether the episode ended
    step_result<state_type> step(const action_type & a)
    {
        step_result<state_type> out;
        out.reward = a == 0 ? 1 : 0;
        at_++;
        out.next = at_;
        out.terminated = at_ == n_;
        return out;
    }

    /// @return the legal actions: 0 and 1
    std::vector<action_type> actions(const state_type &) const
    {
        return {0, 1};
    }

    /**
     * @return the value, from state @param t, of the policy that takes action 0 with probability
     *         @param p, with the discount rate @param gamma: p (1 + γ + ... + γ^(n-t-1))
     */
    double value(state_type t,
                 double p,
                 double gamma) const
    {
        double sum = 0;
        double discount = 1;
        for (unsigned int k = t; k < n_; k++) {
            sum += discount * p;
            discount *= gamma;
        }
        return sum;
    }

private:
    // the number of steps of every episode
    unsigned int n_;
    // the steps taken so far
    unsigned int at_ = 0;
};
} // namespace games
#endif
