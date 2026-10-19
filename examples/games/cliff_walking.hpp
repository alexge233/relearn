#ifndef RELEARN_GAMES_CLIFF_WALKING_HPP
#define RELEARN_GAMES_CLIFF_WALKING_HPP
/**
 * Cliff Walking, as in Sutton & Barto, Example 6.6.
 *
 *     . . . . . . . . . . . .
 *     . . . . . . . . . . . .
 *     . . . . . . . . . . . .
 *     S C C C C C C C C C C G       S start, G goal, C cliff
 *
 * - 12 x 4 grid; moves up, right, down, left; moving into the border stays put.
 * - Every move costs -1. Stepping into the cliff costs -100 and sends the agent
 *   back to the start (the episode goes on). Reaching the goal ends the episode.
 * - The shortest path runs along the cliff edge: 13 moves, return -13.
 *   Q-learning learns it; SARSA, which accounts for its own exploration,
 *   learns the safer path further from the edge.
 */
#include "game.hpp"
#include "grid.hpp"
#include <vector>

namespace games
{
/**
 * @class cliff_walking
 * @brief the Cliff Walking game: walk from the start to the goal without falling off the cliff
 */
class cliff_walking
{
public:
    /// a state is the cell the agent is in
    using state_type  = position;
    /// an action is one of the four moves: up, right, down, left
    using action_type = unsigned int;
    /// size of the grid
    enum : int
    {
        width  = 12,
        height = 4
    };

    /// @return the start: the bottom-left cell
    static position start()
    {
        return {0, height - 1};
    }

    /// @return the goal: the bottom-right cell
    static position goal()
    {
        return {width - 1, height - 1};
    }

    /// @return true if @param p is a cell of the cliff: the bottom row between start and goal
    static bool cliff(const position & p)
    {
        return (p.y == height - 1) &&
               (p.x > 0) &&
               (p.x < width - 1);
    }

    /// @brief start a new episode @return the start
    state_type reset()
    {
        at_ = start();
        return at_;
    }

    /// @brief take the move @param a @return the cell reached, the reward and whether the game ended
    step_result<state_type> step(const action_type & a)
    {
        step_result<state_type> out;
        at_ = moved(at_, a, width, height);
        if (cliff(at_)) {
            // fallen off the cliff: a big penalty, and back to the start
            out.reward = -100;
            at_ = start();
        }
        else {
            // every other move costs -1, so shorter paths are better
            out.reward = -1;
            out.terminated = (at_ == goal());
        }
        out.next = at_;
        return out;
    }

    /// @return the legal moves: all four, in every cell
    std::vector<action_type> actions(const state_type &) const
    {
        return {up, right, down, left};
    }

private:
    // the cell the agent is in
    position at_;
};
} // namespace games
#endif
