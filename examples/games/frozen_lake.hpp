#ifndef RELEARN_GAMES_FROZEN_LAKE_HPP
#define RELEARN_GAMES_FROZEN_LAKE_HPP
/**
 * FrozenLake, with the same rules, maps and numbering as Gymnasium's
 * `FrozenLake-v1`, so agents can later be compared on both.
 *
 *     S F F F        S start, F frozen (safe), H hole, G goal
 *     F H F H
 *     F F F H
 *     H F F G
 *
 * - States are cell numbers, row * columns + column (as in Gymnasium).
 * - Actions: 0 left, 1 down, 2 right, 3 up (as in Gymnasium).
 * - Slippery ice: the agent moves in the chosen direction a third of the time,
 *   and slides to either side of it a third of the time each. Moving into the
 *   border stays put.
 * - Reaching G is worth +1 and ends the episode; falling into H ends it with 0.
 * - Gymnasium stops an episode after 100 moves (4x4) or 200 moves (8x8):
 *   use `step_limit()` with `games::play`.
 */
#include "game.hpp"
#include <random>
#include <string>
#include <vector>

namespace games
{
/**
 * @class frozen_lake
 * @brief the FrozenLake game: cross the slippery ice to the goal without falling into a hole
 */
class frozen_lake
{
public:
    /// a state is the number of the cell the agent is in
    using state_type  = unsigned int;
    /// an action is one of the four moves, numbered as in Gymnasium
    using action_type = unsigned int;
    /// the four moves, numbered as in Gymnasium
    enum : action_type
    {
        left  = 0,
        down  = 1,
        right = 2,
        up    = 3
    };

    /// @return Gymnasium's 4x4 map
    static std::vector<std::string> map_4x4()
    {
        return {"SFFF",
                "FHFH",
                "FFFH",
                "HFFG"};
    }

    /// @return Gymnasium's 8x8 map
    static std::vector<std::string> map_8x8()
    {
        return {"SFFFFFFF",
                "FFFFFFFF",
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFG"};
    }

    /**
     * @brief a new game
     * @param map gives the letter of each cell, one string per row
     * @param slippery selects the slippery ice of Gymnasium
     * @param seed seeds the generator of the slides
     */
    explicit frozen_lake(std::vector<std::string> map = map_4x4(),
                         bool slippery = true,
                         unsigned int seed = 0)
    : map_(std::move(map)),
      slippery_(slippery),
      prng_(seed)
    {
        rows_ = static_cast<int>(map_.size());
        cols_ = static_cast<int>(map_.front().size());
        // find the start on the map
        for (int r = 0; r < rows_; r++) {
            for (int c = 0; c < cols_; c++) {
                if (map_[r][c] == 'S') {
                    start_ = cell(r, c);
                }
            }
        }
    }

    /// @return the number of rows of the map
    int rows() const
    {
        return rows_;
    }

    /// @return the number of columns of the map
    int columns() const
    {
        return cols_;
    }

    /// @return the number of cells, which is the number of states
    unsigned int states() const
    {
        return static_cast<unsigned int>(rows_ * cols_);
    }

    /// @return Gymnasium's limit on the number of moves of an episode
    std::size_t step_limit() const
    {
        return rows_ * cols_ <= 16 ? 100 : 200;
    }

    /// @return the map letter of cell @param s
    char tile(state_type s) const
    {
        return map_[s / cols_][s % cols_];
    }

    /// @brief start a new episode @return the start
    state_type reset()
    {
        at_ = start_;
        return at_;
    }

    /// @brief put the agent on cell @param s (for tests, or to start elsewhere)
    void place(state_type s)
    {
        at_ = s;
    }

    /// @brief take the move @param a @return the cell reached, the reward and whether the game ended
    step_result<state_type> step(const action_type & a)
    {
        action_type actual = a;
        // slippery ice: the move turns left, stays, or turns right, a third of the time each
        if (slippery_) {
            std::uniform_int_distribution<int> slide(-1, 1);
            actual = static_cast<action_type>((static_cast<int>(a) + 4 + slide(prng_)) % 4);
        }
        at_ = moved(at_, actual);
        // the goal is worth +1; the goal and the holes end the episode
        step_result<state_type> out;
        out.next       = at_;
        out.reward     = tile(at_) == 'G' ? 1 : 0;
        out.terminated = tile(at_) == 'G' || tile(at_) == 'H';
        return out;
    }

    /// @return the legal moves: all four, in every cell
    std::vector<action_type> actions(const state_type &) const
    {
        return {left, down, right, up};
    }

    /// @return the cell reached from @param s by moving in direction @param a (no slipping)
    state_type moved(state_type s,
                     action_type a) const
    {
        int r = static_cast<int>(s) / cols_;
        int c = static_cast<int>(s) % cols_;
        // take one step in the direction of the move; the border keeps the agent in
        switch (a) {
            case left:
                c = c > 0 ? c - 1 : 0;
                break;
            case down:
                r = r < rows_ - 1 ? r + 1 : rows_ - 1;
                break;
            case right:
                c = c < cols_ - 1 ? c + 1 : cols_ - 1;
                break;
            case up:
                r = r > 0 ? r - 1 : 0;
                break;
            default:
                break;
        }
        return cell(r, c);
    }

private:
    /// @return the number of the cell in row @param r and column @param c
    state_type cell(int r,
                    int c) const
    {
        return static_cast<state_type>(r * cols_ + c);
    }

    // the letter of each cell, one string per row
    std::vector<std::string> map_;
    // true for slippery ice
    bool slippery_;
    // the generator of the slides
    std::mt19937 prng_;
    // the size of the map
    int rows_ = 0;
    int cols_ = 0;
    // the start cell, and the cell the agent is in
    state_type start_ = 0;
    state_type at_    = 0;
};
} // namespace games
#endif
