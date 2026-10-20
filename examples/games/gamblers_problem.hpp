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
 * - `model()` gives the rules as a model, p(s', r | s, a), for dynamic programming.
 */
#include "game.hpp"
#include <algorithm>
#include <random>
#include <vector>

namespace games
{
/**
 * @class gamblers_problem
 * @brief bet on coin flips until the capital reaches the goal, or runs out
 */
class gamblers_problem
{
public:
    /// a state is the capital
    using state_type  = unsigned int;
    /// an action is the stake
    using action_type = unsigned int;
    /// the rules as a model, for dynamic programming
    using model_type  = relearn::finite_mdp<relearn::state<state_type>,
                                            relearn::action<action_type>>;

    /**
     * @brief create the game
     * @param p_heads is the probability of heads, which wins the stake
     * @param start_capital is the capital each episode starts with;
     *        0 starts each episode with a random capital
     * @param goal is the capital to reach
     * @param seed seeds the coin and the random starting capital
     */
    explicit gamblers_problem(double p_heads = 0.4,
                              unsigned int start_capital = 0,
                              unsigned int goal = 100,
                              unsigned int seed = 0)
    : p_(p_heads),
      start_(start_capital),
      goal_(goal),
      prng_(seed)
    {}

    /// @return the capital to reach
    unsigned int goal() const
    {
        return goal_;
    }

    /// @return the probability of heads
    double p_heads() const
    {
        return p_;
    }

    /// @brief start a new episode @return the starting capital
    state_type reset()
    {
        if (start_ > 0) {
            capital_ = start_;
        }
        else {
            // a random capital, so that every capital gets visited
            std::uniform_int_distribution<unsigned int> any(1, goal_ - 1);
            capital_ = any(prng_);
        }
        return capital_;
    }

    /// @brief bet @param stake on a coin flip @return the capital reached, the reward and whether the game ended
    step_result<state_type> step(const action_type & stake)
    {
        // heads wins the stake, tails loses it
        std::bernoulli_distribution heads(p_);
        if (heads(prng_)) {
            capital_ += stake;
        }
        else {
            capital_ -= stake;
        }
        // reaching the goal is worth +1; the game ends at the goal or when the money runs out
        step_result<state_type> out;
        out.next = capital_;
        out.reward = capital_ >= goal_ ? 1 : 0;
        out.terminated = capital_ == 0 || capital_ >= goal_;
        return out;
    }

    /// @return the legal stakes for @param capital: 1 ... min(capital, goal - capital)
    std::vector<action_type> actions(const state_type & capital) const
    {
        std::vector<action_type> stakes;
        for (unsigned int s = 1; s <= std::min(capital, goal_ - capital); s++) {
            stakes.push_back(s);
        }
        return stakes;
    }

    /**
     * @return the rules as a model, p(s', r | s, a): the capitals 0 ... goal (0 and the goal are
     *         terminal), and each stake leads to a win with probability `p_heads`, or a loss
     */
    model_type model() const
    {
        model_type m;
        for (state_type capital = 0; capital <= goal_; capital++) {
            m.states.push_back(relearn::state<state_type>(capital));
        }
        // the stakes 1 ... min(capital, goal - capital): none at 0 and at the goal
        const unsigned int goal = goal_;
        m.actions = [goal](const relearn::state<state_type> & s) {
            std::vector<relearn::action<action_type>> stakes;
            const unsigned int capital = s.trait();
            for (unsigned int stake = 1; stake <= std::min(capital, goal - capital); stake++) {
                stakes.push_back(relearn::action<action_type>(stake));
            }
            return stakes;
        };
        // heads wins the stake (reaching the goal is worth +1), tails loses it
        const double p = p_;
        m.outcomes = [goal, p](const relearn::state<state_type> & s,
                               const relearn::action<action_type> & a) {
            const unsigned int capital = s.trait();
            const unsigned int stake   = a.trait();
            relearn::state<state_type> won(capital + stake);
            relearn::state<state_type> lost(capital - stake);
            return std::vector<model_type::outcome> {{won, capital + stake >= goal ? 1.0 : 0.0, p},
                                                     {lost, 0.0, 1.0 - p}};
        };
        return m;
    }

private:
    // probability of heads
    double p_;
    // starting capital, 0 for a random one
    unsigned int start_;
    // the capital to reach
    unsigned int goal_;
    // the coin, and the random starting capital
    std::mt19937 prng_;
    // the current capital
    unsigned int capital_ = 0;
};
} // namespace games
#endif
