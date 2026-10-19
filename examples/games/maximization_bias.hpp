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
class maximization_bias
{
public:
    using state_type  = unsigned int;
    using action_type = unsigned int;
    enum : state_type { A = 0, B = 1, end = 2 };
    enum : action_type { right = 0, left = 1 };   // the actions of A; B's are 0 ... k-1

    explicit maximization_bias(unsigned int actions_in_b = 10, unsigned int seed = 0)
    : k_(actions_in_b), prng_(seed)
    {}

    state_type reset()
    {
        at_ = A;
        return at_;
    }

    step_result<state_type> step(const action_type & a)
    {
        step_result<state_type> out;
        if (at_ == A && a == left) {
            at_ = B;
        }
        else if (at_ == A) {
            at_ = end;
            out.terminated = true;
        }
        else {
            std::normal_distribution<double> reward(-0.1, 1.0);
            out.reward = reward(prng_);
            at_ = end;
            out.terminated = true;
        }
        out.next = at_;
        return out;
    }

    std::vector<action_type> actions(const state_type & s) const
    {
        if (s == A) return {right, left};
        std::vector<action_type> all(k_);
        for (unsigned int a = 0; a < k_; a++) all[a] = a;
        return all;
    }

private:
    unsigned int k_;
    std::mt19937 prng_;
    state_type at_ = A;
};
} // namespace games
#endif
