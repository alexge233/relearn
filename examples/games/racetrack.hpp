#ifndef RELEARN_GAMES_RACETRACK_HPP
#define RELEARN_GAMES_RACETRACK_HPP
/**
 * The racetrack of Sutton & Barto, Exercise 5.12 and Example 8.6: drive a car round a turn, as fast as you can,
 * without running off the track.
 *
 * - The track is a grid: '#' is off the track, '.' on it, 'S' the start line and 'F' the finish line.
 * - The state is the car's cell and its velocity, (vx, vy): vx to the right and vy upwards, each 0 ... 4.
 * - Each action changes each velocity component by -1, 0 or +1: 9 actions, those that keep both components
 *   in 0 ... 4 and do not stop the car (both 0) being legal. The book lets the car stand still on the start line;
 *   here it must always move, and it is only at rest when it is put back there.
 * - The car then moves by its velocity. If its path crosses the finish line, the episode ends; if it leaves
 *   the track first, the car is put back on a random cell of the start line, at rest, and the episode goes on.
 *   The path is checked at 4 points per cell of its length, each rounded to the nearest cell (halves away from 0).
 * - Every step costs -1, so the return is minus the number of steps to the finish.
 * - With probability 0.1 (if `noisy`), the velocity does not change at all, whatever the action.
 *
 * The book's track (Figure 5.5) is not given as text, so the default track here is drawn after it: a long climb
 * up from the start line at the bottom, then a turn to the right, to the finish line on the right edge.
 * - States are numbered (cell * 25 + vx * 5 + vy), and `finished()` is the one terminal state.
 * - `model()` gives the rules as a model, p(s', r | s, a), for dynamic programming.
 */
#include "game.hpp"
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace games
{
/**
 * @class racetrack
 * @brief the racetrack: from the start line to the finish line, in as few steps as possible
 */
class racetrack
{
public:
    /// a state is a number: the cell (row * columns + column) and the velocity, cell * 25 + vx * 5 + vy
    using state_type  = unsigned int;
    /// an action is the change of velocity, (dvx + 1) * 3 + (dvy + 1)
    using action_type = unsigned int;
    /// the rules as a model, for dynamic programming
    using model_type  = relearn::finite_mdp<relearn::state<state_type>,
                                            relearn::action<action_type>>;

    /// @return the default track, drawn after the left one of the book's Figure 5.5
    static std::vector<std::string> default_track()
    {
        return {"###.............F",
                "###.............F",
                "###.............F",
                "##..............F",
                "##..............F",
                "##..............F",
                "#..........######",
                "#.........#######",
                "#.........#######",
                "#.........#######",
                "#.........#######",
                "#.........#######",
                "#.........#######",
                "#.........#######",
                "..........#######",
                ".........########",
                ".........########",
                ".........########",
                ".........########",
                ".........########",
                ".........########",
                ".........########",
                "#........########",
                "#........########",
                "#........########",
                "#........########",
                "#........########",
                "#........########",
                "#........########",
                "##.......########",
                "##.......########",
                "###SSSSSS########"};
    }

    /**
     * @brief a new race
     * @param track gives the letter of each cell, one string per row
     * @param noisy is true for the book's noise: with probability 0.1 the velocity does not change
     * @param seed seeds the noise and the start cells
     */
    explicit racetrack(std::vector<std::string> track = default_track(),
                       bool noisy = true,
                       unsigned int seed = 0)
    : track_(std::move(track)),
      noisy_(noisy),
      prng_(seed)
    {
        rows_ = static_cast<int>(track_.size());
        cols_ = static_cast<int>(track_.front().size());
        for (int r = 0; r < rows_; r++) {
            for (int c = 0; c < cols_; c++) {
                if (track_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] == 'S') {
                    starts_.push_back(static_cast<state_type>(r * cols_ + c));
                }
            }
        }
    }

    /// @return the state of the car in cell (@param row, @param column), with velocity (@param vx, @param vy)
    state_type state_of(int row,
                        int column,
                        int vx,
                        int vy) const
    {
        return static_cast<state_type>(((row * cols_ + column) * 5 + vx) * 5 + vy);
    }

    /// @return the number of rows of the track
    int rows() const
    {
        return rows_;
    }

    /// @return the number of columns of the track
    int columns() const
    {
        return cols_;
    }

    /// @return the terminal state: the car has crossed the finish line
    state_type finished() const
    {
        return static_cast<state_type>(rows_ * cols_ * 25);
    }

    /// @return the row of the car in state @param s
    int row(state_type s) const
    {
        return static_cast<int>(s / 25) / cols_;
    }

    /// @return the column of the car in state @param s
    int column(state_type s) const
    {
        return static_cast<int>(s / 25) % cols_;
    }

    /// @return the horizontal velocity of the car in state @param s
    int vx(state_type s) const
    {
        return static_cast<int>(s % 25) / 5;
    }

    /// @return the vertical velocity of the car in state @param s
    int vy(state_type s) const
    {
        return static_cast<int>(s % 5);
    }

    /// @return the letter of the cell in row @param r and column @param c, '#' outside the track
    char tile(int r,
              int c) const
    {
        if (r < 0 || r >= rows_ || c < 0 || c >= cols_) {
            return '#';
        }
        return track_[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
    }

    /// @return the states the car starts from: every cell of the start line, at rest
    std::vector<state_type> starts() const
    {
        std::vector<state_type> states;
        for (auto cell : starts_) {
            states.push_back(cell * 25);
        }
        return states;
    }

    /// @brief start a new race @return the car at rest on a random cell of the start line
    state_type reset()
    {
        std::uniform_int_distribution<std::size_t> pick(0, starts_.size() - 1);
        at_ = starts_[pick(prng_)] * 25;
        return at_;
    }

    /// @brief put the car in state @param s (for tests, or to start elsewhere)
    void place(state_type s)
    {
        at_ = s;
    }

    /// @return the legal actions of state @param s: those that keep both components in 0 ... 4, and the car moving
    std::vector<action_type> actions(const state_type & s) const
    {
        std::vector<action_type> legal;
        if (s == finished()) {
            return legal;
        }
        for (int dvx = -1; dvx <= 1; dvx++) {
            for (int dvy = -1; dvy <= 1; dvy++) {
                const int x = vx(s) + dvx;
                const int y = vy(s) + dvy;
                if (x >= 0 && x <= 4 && y >= 0 && y <= 4 && (x != 0 || y != 0)) {
                    legal.push_back(static_cast<action_type>((dvx + 1) * 3 + (dvy + 1)));
                }
            }
        }
        return legal;
    }

    /**
     * @brief where the car goes from state @param s with velocity (@param x, @param y), without the noise
     * @return the state reached: `finished()` if the path crosses the finish line, or a state at rest on the start
     *         line if it leaves the track first (@param crashed is then true, and the start cell is the one drawn
     *         by @param start, an index into the start line)
     */
    state_type drive(state_type s,
                     int x,
                     int y,
                     std::size_t start,
                     bool & crashed) const
    {
        crashed = false;
        const int r = row(s);
        const int c = column(s);
        // the cells along the path, the finish line first: the car finishes if it gets there before leaving the track
        const int steps = 4 * std::max(std::max(x, y), 1);
        for (int k = 1; k <= steps; k++) {
            const int rr = static_cast<int>(std::lround(r - static_cast<double>(y) * k / steps));
            const int cc = static_cast<int>(std::lround(c + static_cast<double>(x) * k / steps));
            const char t = tile(rr, cc);
            if (t == 'F') {
                return finished();
            }
            if (t == '#') {
                crashed = true;
                return starts_[start] * 25;
            }
        }
        return state_of(r - y, c + x, x, y);
    }

    /// @brief take the action @param a @return the state reached, the reward (-1) and whether the race ended
    step_result<state_type> step(const action_type & a)
    {
        int x = vx(at_);
        int y = vy(at_);
        // with probability 0.1 the velocity does not change
        std::bernoulli_distribution slip(noisy_ ? 0.1 : 0.0);
        if (!slip(prng_)) {
            x += static_cast<int>(a / 3) - 1;
            y += static_cast<int>(a % 3) - 1;
        }
        std::uniform_int_distribution<std::size_t> pick(0, starts_.size() - 1);
        bool crashed = false;
        at_ = drive(at_, x, y, pick(prng_), crashed);
        step_result<state_type> out;
        out.next = at_;
        out.reward = -1;
        out.terminated = at_ == finished();
        return out;
    }

    /// @return every state of the model: every cell on the track (but the finish line) at every velocity, and `finished()`
    std::vector<state_type> states() const
    {
        std::vector<state_type> all;
        for (int r = 0; r < rows_; r++) {
            for (int c = 0; c < cols_; c++) {
                if (tile(r, c) == '.' || tile(r, c) == 'S') {
                    for (int x = 0; x <= 4; x++) {
                        for (int y = 0; y <= 4; y++) {
                            all.push_back(state_of(r, c, x, y));
                        }
                    }
                }
            }
        }
        all.push_back(finished());
        return all;
    }

    /**
     * @return the rules as a model, p(s', r | s, a): the velocity changes as asked (with probability 0.9, if noisy)
     *         or not at all (0.1), the car drives, and a crash puts it at rest on each cell of the start line alike
     */
    model_type model() const
    {
        model_type m;
        for (auto s : states()) {
            m.states.push_back(relearn::state<state_type>(s));
        }
        auto self = *this;
        m.actions = [self](const relearn::state<state_type> & s) {
            std::vector<relearn::action<action_type>> moves;
            for (auto a : self.actions(s.trait())) {
                moves.push_back(relearn::action<action_type>(a));
            }
            return moves;
        };
        m.outcomes = [self](const relearn::state<state_type> & s,
                            const relearn::action<action_type> & a) {
            std::vector<model_type::outcome> outcomes;
            const int x = self.vx(s.trait());
            const int y = self.vy(s.trait());
            // the velocity as asked, and unchanged
            const double asked = self.noisy_ ? 0.9 : 1.0;
            std::vector<std::pair<std::pair<int,int>, double>> velocities = {
                {{x + static_cast<int>(a.trait() / 3) - 1, y + static_cast<int>(a.trait() % 3) - 1}, asked}};
            if (self.noisy_) {
                velocities.push_back({{x, y}, 0.1});
            }
            for (const auto & v : velocities) {
                bool crashed = false;
                const auto next = self.drive(s.trait(), v.first.first, v.first.second, 0, crashed);
                if (crashed) {
                    // back to the start line, each cell alike
                    for (std::size_t i = 0; i < self.starts_.size(); i++) {
                        outcomes.push_back({relearn::state<state_type>(self.starts_[i] * 25), -1.0,
                                            v.second / static_cast<double>(self.starts_.size())});
                    }
                }
                else {
                    outcomes.push_back({relearn::state<state_type>(next), -1.0, v.second});
                }
            }
            return outcomes;
        };
        return m;
    }

private:
    // the letter of each cell, one string per row
    std::vector<std::string> track_;
    // true for the noise: with probability 0.1 the velocity does not change
    bool noisy_;
    // the noise, and the start cells
    std::mt19937 prng_;
    // the size of the track
    int rows_ = 0;
    int cols_ = 0;
    // the cells of the start line
    std::vector<state_type> starts_;
    // the state of the car
    state_type at_ = 0;
};
} // namespace games
#endif
