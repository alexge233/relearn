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
 * - Optionally (off by default), a maze can be `scale`d up, as in Example 8.4: each cell
 *   becomes a block of scale x scale cells, so the maze keeps its shape at a finer
 *   resolution. The start and the goal are the top-left cells of their blocks.
 *   The maze of Example 8.1 at scale 2 is 18 x 12, with 188 open cells:
 *
 *     . . . . . . . . . . . . . . # # G .
 *     . . . . . . . . . . . . . . # # . .
 *     . . . . # # . . . . . . . . # # . .
 *     . . . . # # . . . . . . . . # # . .
 *     S . . . # # . . . . . . . . # # . .
 *     . . . . # # . . . . . . . . # # . .
 *     . . . . # # . . . . . . . . . . . .
 *     . . . . # # . . . . . . . . . . . .
 *     . . . . . . . . . . # # . . . . . .
 *     . . . . . . . . . . # # . . . . . .
 *     . . . . . . . . . . . . . . . . . .
 *     . . . . . . . . . . . . . . . . . .
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
    /// size of the grid, at scale 1 (see `columns()` and `rows()` for a scaled maze)
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
     * @param scale makes each cell a block of scale x scale cells (1, the default, is the book's maze)
     */
    explicit dyna_maze(layout_type layout = maze,
                       unsigned long change_after = 1000,
                       int scale = 1)
    : layout_(layout),
      change_after_(change_after),
      scale_(scale)
    {}

    /// @return the start: on the left in the maze, at the bottom in the changing mazes
    position start() const
    {
        position at = layout_ == maze ? position{0, 2} : position{3, 5};
        return {at.x * scale_, at.y * scale_};
    }

    /// @return the goal: the top-right cell (of a scaled maze, the top-left cell of the top-right block)
    position goal() const
    {
        return {8 * scale_, 0};
    }

    /// @return the number of columns: `width` times the scale
    int columns() const
    {
        return width * scale_;
    }

    /// @return the number of rows: `height` times the scale
    int rows() const
    {
        return height * scale_;
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
        // the cell of the book's maze that p is part of (p itself, unless the maze is scaled)
        const position cell = {p.x / scale_, p.y / scale_};
        // the maze of Example 8.1: three fixed walls
        if (layout_ == maze) {
            return (cell.x == 2 && cell.y >= 1 && cell.y <= 3) ||
                   (cell.x == 5 && cell.y == 4) ||
                   (cell.x == 7 && cell.y <= 2);
        }
        // the changing mazes have a single wall, along row 3
        if (cell.y != 3) {
            return false;
        }
        // blocking: the gap moves from the right end to the left end
        if (layout_ == blocking) {
            return changed() ? cell.x >= 1 : cell.x <= 7;
        }
        // shortcut: the left gap stays, and a gap opens on the right
        return changed() ? (cell.x >= 1 && cell.x <= 7) : cell.x >= 1;
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
        position to = moved(at_, a, columns(), rows());
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
        std::vector<std::string> lines(rows(), std::string(columns(), '.'));
        for (int y = 0; y < rows(); y++) {
            for (int x = 0; x < columns(); x++) {
                if (wall({x, y})) {
                    lines[y][x] = '#';
                }
            }
        }
        lines[start().y][start().x] = 'S';
        lines[goal().y][goal().x]   = 'G';
        return lines;
    }

private:
    // which maze this is
    layout_type layout_;
    // moves after which a changing maze changes
    unsigned long change_after_;
    // each cell of the book's maze is a block of scale_ x scale_ cells
    int scale_;
    // moves taken since the game was created
    unsigned long moves_ = 0;
    // the cell the agent is in
    position at_;
};
} // namespace games
#endif
