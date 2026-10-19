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
/**
 * @class windy_gridworld
 * @brief the Windy Gridworld game: reach the goal while the wind pushes the agent up
 */
class windy_gridworld
{
public:
    /// a state is the cell the agent is in
    using state_type  = position;
    /// an action is one of the four moves: up, right, down, left
    using action_type = unsigned int;
    /// size of the grid
    enum : int
    {
        width  = 10,
        height = 7
    };

    /**
     * @brief a new game
     * @param stochastic_wind selects Exercise 6.10's random wind
     * @param seed seeds the generator of the random wind
     */
    explicit windy_gridworld(bool stochastic_wind = false,
                             unsigned int seed = 0)
    : stochastic_(stochastic_wind),
      prng_(seed)
    {}

    /// @return the start: column 0, row 3
    static position start()
    {
        return {0, 3};
    }

    /// @return the goal: column 7, row 3
    static position goal()
    {
        return {7, 3};
    }

    /// @return the upward wind of @param column
    static int wind(int column)
    {
        static const int strength[width] = {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};
        return strength[column];
    }

    /// @brief start a new episode @return the start
    state_type reset()
    {
        at_ = start();
        return at_;
    }

    /// @brief put the agent on cell @param at (for tests, or to start elsewhere)
    void place(const position & at)
    {
        at_ = at;
    }

    /// @brief take the move @param a @return the cell reached, the reward and whether the game ended
    step_result<state_type> step(const action_type & a)
    {
        // the wind of the column the agent moves from
        int push = wind(at_.x);
        // random wind: where it blows, one cell stronger or weaker a third of the time each
        if (stochastic_ && push > 0) {
            std::uniform_int_distribution<int> vary(-1, 1);
            push += vary(prng_);
        }
        // move, then get pushed up by the wind, staying inside the grid
        position to = moved(at_, a, width, height);
        to.y -= push;
        if (to.y < 0) {
            to.y = 0;
        }
        if (to.y >= height) {
            to.y = height - 1;
        }
        at_ = to;
        // every move costs -1, and reaching the goal ends the episode
        step_result<state_type> out;
        out.next       = at_;
        out.reward     = -1;
        out.terminated = (at_ == goal());
        return out;
    }

    /// @return the legal moves: all four, in every cell
    std::vector<action_type> actions(const state_type &) const
    {
        return {up, right, down, left};
    }

private:
    // true for Exercise 6.10's random wind
    bool stochastic_;
    // the generator of the random wind
    std::mt19937 prng_;
    // the cell the agent is in
    position at_;
};
} // namespace games
#endif
