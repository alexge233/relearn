#ifndef RELEARN_GAMES_DYNA_MAZE_HPP
#define RELEARN_GAMES_DYNA_MAZE_HPP
/**
 * The Dyna maze of Sutton & Barto, Example 8.1 (Figure 8.2), and two mazes that
 * change during learning, after Figures 8.4 (blocking) and 8.5 (shortcut).
 *
 *     maze               blocking             shortcut
 *     . . . . . . . # G  . . . . . . . . G    . . . . . . . . G
 *     . . # . . . . # .  . . . . . . . . .    . . . . . . . . .
 *     S . # . . . . # .  . . . . . . . . .    . . . . . . . . .
 *     . . # . . . . . .  # # # # # # # # .    . # # # # # # # #   <- this wall
 *     . . . . . # . . .  . . . . . . . . .    . . . . . . . . .      changes
 *     . . . . . . . . .  . . . S . . . . .    . . . S . . . . .
 *
 * - 9 x 6 grid; moves up, right, down, left; walls and the border keep the agent in place.
 * - Reaching G is worth +1 and ends the episode; every other move is worth 0,
 *   so a discount (0.95 in the book) is what makes shorter paths better.
 * - The blocking maze closes its right-hand gap and opens one on the left after
 *   `change_after` moves; the shortcut maze keeps its left gap and opens a new,
 *   shorter one on the right.
 */
#include "game.hpp"
#include "grid.hpp"
#include <string>
#include <vector>

namespace games
{
/**
 * @class dyna_maze
 * @brief a maze to cross from S to G; the blocking and shortcut mazes change while the agent learns
 */
class dyna_maze
{
public:
    /// a state is the cell the agent is in
    using state_type  = position;
    /// an action is one of the four moves: up, right, down, left
    using action_type = unsigned int;
    /// size of the grid
    enum : int
    {
        width  = 9,
        height = 6
    };
    /// the three mazes
    enum layout_type
    {
        maze,
        blocking,
        shortcut
    };

    /**
     * @brief create a maze
     * @param layout is the maze to play
     * @param change_after is the number of moves after which a changing maze changes
     */
    explicit dyna_maze(layout_type layout = maze,
                       unsigned long change_after = 1000)
    : layout_(layout),
      change_after_(change_after)
    {}

    /// @return the start: on the left in the maze, at the bottom in the changing mazes
    position start() const
    {
        return layout_ == maze ? position{0, 2} : position{3, 5};
    }

    /// @return the goal: the top-right cell
    static position goal()
    {
        return {8, 0};
    }

    /// @return true after a changing maze has changed
    bool changed() const
    {
        return layout_ != maze && moves_ >= change_after_;
    }

    /// @return the moves taken since the game was created
    unsigned long moves() const
    {
        return moves_;
    }

    /// @return true if cell @param p is a wall, in the current layout
    bool wall(const position & p) const
    {
        // the maze of Example 8.1: three fixed walls
        if (layout_ == maze) {
            return (p.x == 2 && p.y >= 1 && p.y <= 3) ||
                   (p.x == 5 && p.y == 4) ||
                   (p.x == 7 && p.y <= 2);
        }
        // the changing mazes have a single wall, along row 3
        if (p.y != 3) {
            return false;
        }
        // blocking: the gap moves from the right end to the left end
        if (layout_ == blocking) {
            return changed() ? p.x >= 1 : p.x <= 7;
        }
        // shortcut: the left gap stays, and a gap opens on the right
        return changed() ? (p.x >= 1 && p.x <= 7) : p.x >= 1;
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
        // count every move: the changing mazes change after a number of moves
        moves_++;
        // a move into a wall keeps the agent in place
        position to = moved(at_, a, width, height);
        if (!wall(to)) {
            at_ = to;
        }
        // only reaching the goal is rewarded
        step_result<state_type> out;
        out.next = at_;
        out.terminated = at_ == goal();
        out.reward = out.terminated ? 1 : 0;
        return out;
    }

    /// @return the legal moves: all four, in every cell
    std::vector<action_type> actions(const state_type &) const
    {
        return {up, right, down, left};
    }

    /// @return the maze as text, one string per row: '#' walls, 'S' start, 'G' goal
    std::vector<std::string> draw() const
    {
        std::vector<std::string> rows(height, std::string(width, '.'));
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (wall({x, y})) {
                    rows[y][x] = '#';
                }
            }
        }
        rows[start().y][start().x] = 'S';
        rows[goal().y][goal().x]   = 'G';
        return rows;
    }

private:
    // which maze this is
    layout_type layout_;
    // moves after which a changing maze changes
    unsigned long change_after_;
    // moves taken since the game was created
    unsigned long moves_ = 0;
    // the cell the agent is in
    position at_;
};
} // namespace games
#endif
