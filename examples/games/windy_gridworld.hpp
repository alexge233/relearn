#ifndef RELEARN_GAMES_WINDY_GRIDWORLD_HPP
#define RELEARN_GAMES_WINDY_GRIDWORLD_HPP
/**
 * Windy Gridworld, as in Sutton & Barto, Example 6.5 and Exercise 6.10.
 *
 *     . . . . . . . . . .
 *     . . . . . . . . . .
 *     . . . . . . . . . .
 *     S . . . . . . G . .        S start, G goal
 *     . . . . . . . . . .
 *     . . . . . . . . . .
 *     . . . . . . . . . .
 *     0 0 0 1 1 1 2 2 1 0        upward wind of each column
 *
 * - 10 x 7 grid; moves up, right, down, left; the grid's border keeps the agent in.
 * - After each move the agent is also pushed up by the wind of the column it
 *   moved from. Every move costs -1; reaching the goal ends the episode.
 * - The shortest path takes 15 moves.
 * - Stochastic wind (Exercise 6.10): where there is wind, it blows one cell
 *   stronger or weaker a third of the time each.
 */
#include "game.hpp"
#include "grid.hpp"
#include <random>
#include <vector>

namespace games
{
class windy_gridworld
{
public:
    using state_type  = position;
    using action_type = unsigned int;
    enum : int { width = 10, height = 7 };

    /// @param stochastic_wind selects Exercise 6.10's random wind
    explicit windy_gridworld(bool stochastic_wind = false, unsigned int seed = 0)
    : stochastic_(stochastic_wind), prng_(seed)
    {}

    static position start() { return {0, 3}; }
    static position goal()  { return {7, 3}; }
    static int wind(int column)
    {
        static const int strength[width] = {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};
        return strength[column];
    }

    state_type reset()
    {
        at_ = start();
        return at_;
    }

    /// put the agent on cell @param at (for tests, or to start elsewhere)
    void place(const position & at)
    {
        at_ = at;
    }

    step_result<state_type> step(const action_type & a)
    {
        int push = wind(at_.x);
        if (stochastic_ && push > 0) {
            std::uniform_int_distribution<int> vary(-1, 1);
            push += vary(prng_);
        }
        position to = moved(at_, a, width, height);
        to.y -= push;
        if (to.y < 0) to.y = 0;
        if (to.y >= height) to.y = height - 1;
        at_ = to;
        step_result<state_type> out;
        out.next = at_;
        out.reward = -1;
        out.terminated = (at_ == goal());
        return out;
    }

    std::vector<action_type> actions(const state_type &) const
    {
        return {up, right, down, left};
    }

private:
    bool stochastic_;
    std::mt19937 prng_;
    position at_;
};
} // namespace games
#endif
