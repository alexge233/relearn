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
class cliff_walking
{
public:
    using state_type  = position;
    using action_type = unsigned int;
    enum : int { width = 12, height = 4 };

    static position start() { return {0, height - 1}; }
    static position goal()  { return {width - 1, height - 1}; }
    static bool cliff(const position & p) { return p.y == height - 1 && p.x > 0 && p.x < width - 1; }

    state_type reset()
    {
        at_ = start();
        return at_;
    }

    step_result<state_type> step(const action_type & a)
    {
        step_result<state_type> out;
        at_ = moved(at_, a, width, height);
        if (cliff(at_)) {
            out.reward = -100;
            at_ = start();
        }
        else {
            out.reward = -1;
            out.terminated = (at_ == goal());
        }
        out.next = at_;
        return out;
    }

    std::vector<action_type> actions(const state_type &) const
    {
        return {up, right, down, left};
    }

private:
    position at_;
};
} // namespace games
#endif
