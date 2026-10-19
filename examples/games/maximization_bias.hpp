#ifndef RELEARN_GAMES_MAXIMIZATION_BIAS_HPP
#define RELEARN_GAMES_MAXIMIZATION_BIAS_HPP
/**
 * The maximization bias example, as in Sutton & Barto, Example 6.7.
 *
 *     [end] <-- right -- A -- left --> B -- any of k actions --> [end]
 *
 * - From A, "right" ends the episode with reward 0; "left" moves to B with reward 0.
 * - From B, each of the k actions ends the episode with a reward drawn from a
 *   normal distribution with mean -0.1 and variance 1.
 * - Going left is worse on average (-0.1), but some of B's actions look good by
 *   chance, and Q-learning's max over them makes "left" look better than it is.
 *   Double Q-learning removes this bias.
 */
#include "game.hpp"
#include <random>
#include <vector>

namespace games
{
/**
 * @class maximization_bias
 * @brief a two-state game where going left looks better than it is
 */
class maximization_bias
{
public:
    /// a state is A, B or the end
    using state_type  = unsigned int;
    /// an action is right or left in A, and one of 0 ... k-1 in B
    using action_type = unsigned int;
    /// the states
    enum : state_type
    {
        A   = 0,
        B   = 1,
        end = 2
    };
    /// the actions of A; B's are 0 ... k-1
    enum : action_type
    {
        right = 0,
        left  = 1
    };

    /**
     * @brief create the game
     * @param actions_in_b is the number k of actions in B
     * @param seed seeds the random rewards of B
     */
    explicit maximization_bias(unsigned int actions_in_b = 10,
                               unsigned int seed = 0)
    : k_(actions_in_b),
      prng_(seed)
    {}

    /// @brief start a new episode @return state A
    state_type reset()
    {
        at_ = A;
        return at_;
    }

    /// @brief take action @param a @return the state reached, the reward and whether the game ended
    step_result<state_type> step(const action_type & a)
    {
        step_result<state_type> out;
        if (at_ == A && a == left) {
            // left from A leads to B, with no reward
            at_ = B;
        }
        else if (at_ == A) {
            // right from A ends the game, with no reward
            at_ = end;
            out.terminated = true;
        }
        else {
            // any action in B ends the game, with a reward of -0.1 on average
            std::normal_distribution<double> reward(-0.1, 1.0);
            out.reward = reward(prng_);
            at_ = end;
            out.terminated = true;
        }
        out.next = at_;
        return out;
    }

    /// @return the legal actions in @param s: right and left in A, 0 ... k-1 in B
    std::vector<action_type> actions(const state_type & s) const
    {
        if (s == A) {
            return {right, left};
        }
        std::vector<action_type> all(k_);
        for (unsigned int a = 0; a < k_; a++) {
            all[a] = a;
        }
        return all;
    }

private:
    // number of actions in B
    unsigned int k_;
    // draws the rewards of B
    std::mt19937 prng_;
    // the current state
    state_type at_ = A;
};
} // namespace games
#endif
