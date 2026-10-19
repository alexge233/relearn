#ifndef RELEARN_GAMES_GAMBLERS_PROBLEM_HPP
#define RELEARN_GAMES_GAMBLERS_PROBLEM_HPP
/**
 * The gambler's problem, as in Sutton & Barto, Example 4.3.
 *
 * - The state is the gambler's capital, 1 ... goal-1 (goal = 100).
 * - The action is the stake, 1 ... min(capital, goal - capital).
 * - A coin flip with probability `p_heads` (0.4) of heads: heads wins the stake,
 *   tails loses it. Reaching the goal is worth +1, every other step 0; the
 *   episode ends at the goal or when the money runs out.
 * - With no discount, the value of a capital is the probability of reaching
 *   the goal from it. Many stakes are equally good, so the optimal policy is
 *   not unique: compare values, not actions.
 */
#include "game.hpp"
#include <algorithm>
#include <random>
#include <vector>

namespace games
{
class gamblers_problem
{
public:
    using state_type  = unsigned int;    // capital
    using action_type = unsigned int;    // stake

    /// @param start_capital 0 starts each episode with a random capital
    explicit gamblers_problem(double p_heads = 0.4,
                              unsigned int start_capital = 0,
                              unsigned int goal = 100,
                              unsigned int seed = 0)
    : p_(p_heads), start_(start_capital), goal_(goal), prng_(seed)
    {}

    unsigned int goal() const { return goal_; }
    double p_heads() const { return p_; }

    state_type reset()
    {
        if (start_ > 0) {
            capital_ = start_;
        }
        else {
            std::uniform_int_distribution<unsigned int> any(1, goal_ - 1);
            capital_ = any(prng_);
        }
        return capital_;
    }

    step_result<state_type> step(const action_type & stake)
    {
        std::bernoulli_distribution heads(p_);
        if (heads(prng_)) {
            capital_ += stake;
        }
        else {
            capital_ -= stake;
        }
        step_result<state_type> out;
        out.next = capital_;
        out.reward = capital_ >= goal_ ? 1 : 0;
        out.terminated = capital_ == 0 || capital_ >= goal_;
        return out;
    }

    std::vector<action_type> actions(const state_type & capital) const
    {
        std::vector<action_type> stakes;
        for (unsigned int s = 1; s <= std::min(capital, goal_ - capital); s++) {
            stakes.push_back(s);
        }
        return stakes;
    }

private:
    double p_;
    unsigned int start_;
    unsigned int goal_;
    std::mt19937 prng_;
    unsigned int capital_ = 0;
};
} // namespace games
#endif
