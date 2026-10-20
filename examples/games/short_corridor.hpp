#ifndef RELEARN_GAMES_SHORT_CORRIDOR_HPP
#define RELEARN_GAMES_SHORT_CORRIDOR_HPP
/**
 * The short corridor with switched actions, as in Sutton & Barto, Example 13.1.
 *
 *     [S] [x] [ ] [G]          start in S, the goal is G
 *
 * - Three cells before the goal, and two actions, left and right.
 * - In the first and the third cell the actions do what they say (left in the first cell
 *   bumps into the wall, and stays there). In the second cell, x, they are *switched*:
 *   right moves to the left, and left moves to the right.
 * - Every step is worth -1, until the agent reaches G.
 * - The agent **cannot tell the cells apart**: whatever cell it is in, it sees the same state, 0.
 *   So it can only choose how often to go right, the same in every cell, and the best policy is
 *   stochastic: always right gets stuck going back and forth around x, and so does always left, in S.
 *
 * With p the probability of going right, the value of the start (have a look at `start_value`) is
 *
 *     v(S) = (2p - 4) / (p (1 - p))
 *
 * which is best at p = 2 - √2 ≈ 0.5858, where v(S) ≈ -11.657.
 *
 * Optionally (off by default), the agent can be shown which cell it is in: it then sees 0, 1 or 2,
 * and the best policy is the deterministic right, left, right, worth -3.
 */
#include "game.hpp"
#include <cmath>
#include <vector>

namespace games
{
/**
 * @class short_corridor
 * @brief a corridor of three cells, where the second one switches left and right, and all three look the same
 */
class short_corridor
{
public:
    /// a state is what the agent sees: always 0, or the cell 0, 1, 2 when shown
    using state_type  = unsigned int;
    /// an action is `go_left` or `go_right`
    using action_type = unsigned int;
    /// the two actions
    enum : action_type
    {
        go_left  = 0,
        go_right = 1
    };

    /**
     * @brief show the agent which cell it is in (off by default)
     * @param on is true for the agent to see its cell, 0, 1 or 2, rather than always 0
     */
    void show_position(bool on)
    {
        shown_ = on;
    }

    /// @brief start a new episode @return what the agent sees in the first cell
    state_type reset()
    {
        cell_ = 0;
        return observe();
    }

    /// @brief take @param a, `go_left` or `go_right` @return what the agent sees next, the reward -1, and whether it reached G
    step_result<state_type> step(const action_type & a)
    {
        // the second cell switches the actions round
        bool goes_right = (a == go_right);
        if (cell_ == 1) {
            goes_right = !goes_right;
        }
        // move, but left in the first cell bumps into the wall
        if (goes_right) {
            cell_++;
        }
        else if (cell_ > 0) {
            cell_--;
        }
        step_result<state_type> out;
        out.reward = -1;
        out.terminated = (cell_ == 3);
        out.next = out.terminated ? 0 : observe();
        return out;
    }

    /// @return the two actions, left and right, which are legal everywhere
    std::vector<action_type> actions(const state_type &) const
    {
        return {go_left, go_right};
    }

    /// @return the cell the agent is really in: 0, 1, 2 (or 3 once it has reached G)
    unsigned int cell() const
    {
        return cell_;
    }

    /**
     * @return the value of the start, v(S), of the policy which goes right with probability
     *         @param right_first in the first cell, @param right_second in the second, and
     *         @param right_third in the third (each of them above 0 and below 1)
     *
     * With p1, p2, p3 the probabilities of right, the values of the three cells satisfy
     *     v1 = -1 + p1 v2 + (1 - p1) v1
     *     v2 = -1 + p2 v1 + (1 - p2) v3        (right goes back in the second cell)
     *     v3 = -1 + (1 - p3) v2                (right reaches G, worth 0)
     * which give v2 = -(2 - p2 + p2 / p1) / ((1 - p2) p3), and v1 = v2 - 1 / p1.
     */
    static double start_value(double right_first,
                              double right_second,
                              double right_third)
    {
        double second = -(2.0 - right_second + right_second / right_first)
                        / ((1.0 - right_second) * right_third);
        return second - 1.0 / right_first;
    }

    /**
     * @return the value of the start, v(S) = (2p - 4) / (p (1 - p)), of the policy which
     *         goes right with probability @param right = p in every cell (0 < p < 1)
     */
    static double start_value(double right)
    {
        return (2.0 * right - 4.0) / (right * (1.0 - right));
    }

    /// @return the best probability of right when the cells look the same: p = 2 - √2
    static double best_right()
    {
        // dv/dp = 0 where p² - 4p + 2 = 0, and the root between 0 and 1 is 2 - √2
        return 2.0 - std::sqrt(2.0);
    }

private:
    /// @return what the agent sees: its cell if shown, otherwise always 0
    state_type observe() const
    {
        return shown_ ? cell_ : 0;
    }

    // the cell the agent is in: 0, 1, 2, and 3 is G
    unsigned int cell_ = 0;
    // true if the agent sees its cell
    bool shown_ = false;
};
} // namespace games
#endif
