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
class frozen_lake
{
public:
    using state_type  = unsigned int;
    using action_type = unsigned int;
    enum : action_type { left = 0, down = 1, right = 2, up = 3 };

    static std::vector<std::string> map_4x4()
    {
        return {"SFFF", "FHFH", "FFFH", "HFFG"};
    }
    static std::vector<std::string> map_8x8()
    {
        return {"SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF",
                "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"};
    }

    explicit frozen_lake(std::vector<std::string> map = map_4x4(),
                         bool slippery = true,
                         unsigned int seed = 0)
    : map_(std::move(map)), slippery_(slippery), prng_(seed)
    {
        rows_ = static_cast<int>(map_.size());
        cols_ = static_cast<int>(map_.front().size());
        for (int r = 0; r < rows_; r++) {
            for (int c = 0; c < cols_; c++) {
                if (map_[r][c] == 'S') start_ = cell(r, c);
            }
        }
    }

    int rows() const { return rows_; }
    int columns() const { return cols_; }
    unsigned int states() const { return static_cast<unsigned int>(rows_ * cols_); }
    /// Gymnasium's limit on the number of moves of an episode
    std::size_t step_limit() const { return rows_ * cols_ <= 16 ? 100 : 200; }
    /// the map letter of cell @param s
    char tile(state_type s) const { return map_[s / cols_][s % cols_]; }

    state_type reset()
    {
        at_ = start_;
        return at_;
    }

    /// put the agent on cell @param s (for tests, or to start elsewhere)
    void place(state_type s)
    {
        at_ = s;
    }

    step_result<state_type> step(const action_type & a)
    {
        action_type actual = a;
        if (slippery_) {
            std::uniform_int_distribution<int> slide(-1, 1);
            actual = static_cast<action_type>((static_cast<int>(a) + 4 + slide(prng_)) % 4);
        }
        at_ = moved(at_, actual);
        step_result<state_type> out;
        out.next = at_;
        out.reward = tile(at_) == 'G' ? 1 : 0;
        out.terminated = tile(at_) == 'G' || tile(at_) == 'H';
        return out;
    }

    std::vector<action_type> actions(const state_type &) const
    {
        return {left, down, right, up};
    }

    /// the cell reached from @param s by moving in direction @param a (no slipping)
    state_type moved(state_type s, action_type a) const
    {
        int r = static_cast<int>(s) / cols_;
        int c = static_cast<int>(s) % cols_;
        switch (a) {
            case left:  c = c > 0 ? c - 1 : 0; break;
            case down:  r = r < rows_ - 1 ? r + 1 : rows_ - 1; break;
            case right: c = c < cols_ - 1 ? c + 1 : cols_ - 1; break;
            case up:    r = r > 0 ? r - 1 : 0; break;
            default:    break;
        }
        return cell(r, c);
    }

private:
    state_type cell(int r, int c) const { return static_cast<state_type>(r * cols_ + c); }

    std::vector<std::string> map_;
    bool slippery_;
    std::mt19937 prng_;
    int rows_ = 0;
    int cols_ = 0;
    state_type start_ = 0;
    state_type at_ = 0;
};
} // namespace games
#endif
