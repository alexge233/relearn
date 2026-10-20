#ifndef RELEARN_GAMES_SMALL_GRIDWORLD_HPP
#define RELEARN_GAMES_SMALL_GRIDWORLD_HPP
/**
 * The 4 x 4 gridworld of Sutton & Barto, Example 4.1.
 *
 *     [ T]   1    2    3
 *       4    5    6    7          T: the terminal cells
 *       8    9   10   11
 *      12   13   14  [ T]
 *
 * - The states are the cells, numbered 0 ... 15 row by row; the two shaded corners,
 *   0 and 15, are terminal (the book draws one terminal state, twice).
 * - The moves are up, right, down and left; a move off the grid leaves the agent where it is.
 * - Every move is worth -1, until the agent reaches a terminal corner, which ends the episode.
 *   With no discount, the value of a cell is minus the expected number of moves to a corner.
 * - `model()` gives the rules as a model, p(s', r | s, a), for dynamic programming.
 */
#include "game.hpp"
#include "grid.hpp"
#include <random>
#include <vector>

namespace games
{
/**
 * @class small_gridworld
 * @brief walk to either terminal corner of a 4 x 4 grid, paying -1 for every move
 */
class small_gridworld
{
public:
    /// a state is the number of a cell: 0 ... 15, row by row
    using state_type  = unsigned int;
    /// an action is one of the four moves: up, right, down, left
    using action_type = unsigned int;
    /// the rules as a model, for dynamic programming
    using model_type  = relearn::finite_mdp<relearn::state<state_type>,
                                            relearn::action<action_type>>;
    /// size of the grid
    enum : int
    {
        width  = 4,
        height = 4
    };

    /**
     * @brief create the gridworld
     * @param seed seeds the random cell each episode starts from
     */
    explicit small_gridworld(unsigned int seed = 0)
    : prng_(seed)
    {}

    /// @return true if @param s is one of the two terminal corners
    static bool terminal(state_type s)
    {
        return s == 0 || s == 15;
    }

    /// @return the cell reached by taking move @param a in cell @param s
    static state_type moved_from(state_type s,
                                 action_type a)
    {
        // a move off the grid leaves the agent where it is
        position from = {static_cast<int>(s % width), static_cast<int>(s / width)};
        position to = moved(from, a, width, height);
        return static_cast<state_type>(to.y * width + to.x);
    }

    /// @brief start a new episode, in a random cell that is not terminal @return the cell
    state_type reset()
    {
        std::uniform_int_distribution<state_type> any(1, 14);
        at_ = any(prng_);
        return at_;
    }

    /// @brief put the agent in cell @param s (for tests, or to start elsewhere)
    void place(state_type s)
    {
        at_ = s;
    }

    /// @brief take the move @param a @return the cell reached, the reward and whether the game ended
    step_result<state_type> step(const action_type & a)
    {
        // every move is worth -1, and a terminal corner ends the episode
        at_ = moved_from(at_, a);
        step_result<state_type> out;
        out.next = at_;
        out.reward = -1;
        out.terminated = terminal(at_);
        return out;
    }

    /// @return the legal moves: all four, in every cell that is not terminal
    std::vector<action_type> actions(const state_type & s) const
    {
        if (terminal(s)) {
            return {};
        }
        return {up, right, down, left};
    }

    /// @return the rules as a model, p(s', r | s, a): every move leads to one cell for sure, with r = -1
    model_type model() const
    {
        model_type m;
        for (state_type s = 0; s < width * height; s++) {
            m.states.push_back(relearn::state<state_type>(s));
        }
        // the four moves, in every cell that is not terminal
        m.actions = [](const relearn::state<state_type> & s) {
            std::vector<relearn::action<action_type>> moves;
            if (!terminal(s.trait())) {
                for (action_type a : {up, right, down, left}) {
                    moves.push_back(relearn::action<action_type>(a));
                }
            }
            return moves;
        };
        // each move leads to one cell for sure, and is worth -1
        m.outcomes = [](const relearn::state<state_type> & s,
                        const relearn::action<action_type> & a) {
            relearn::state<state_type> next(moved_from(s.trait(), a.trait()));
            return std::vector<model_type::outcome> {{next, -1.0, 1.0}};
        };
        return m;
    }

private:
    // the random cell each episode starts from
    std::mt19937 prng_;
    // the cell the agent is in
    state_type at_ = 1;
};
} // namespace games
#endif
