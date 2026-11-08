#ifndef RELEARN_WINDOW_FROZEN_HPP
#define RELEARN_WINDOW_FROZEN_HPP
/**
 * What the FrozenLake lesson needs on top of the others, with no graphics: the game as a grid, a session that knows a hole
 * from the treasure, and how to say that the ice sent the walker somewhere else.
 */
#include "../games/frozen_lake.hpp"
#include "lesson.hpp"
#include <numeric>
#include <string>
#include <vector>

namespace games
{
namespace lessons
{
/**
 * @class frozen_grid
 * @brief `frozen_lake` as the lessons see a game: its states are `position`s, its moves up, right, down, left
 */
class frozen_grid
{
public:
    using state_type  = position;
    using action_type = unsigned int;

    /// @param seed seeds the slides, @param map is the lake
    explicit frozen_grid(unsigned int seed = 0,
                         std::vector<std::string> map = frozen_lake::map_4x4())
    : lake_(std::move(map), true, seed)
    {}

    /// @return the rows of the lake
    int rows() const
    {
        return lake_.rows();
    }

    /// @return the columns of the lake
    int columns() const
    {
        return lake_.columns();
    }

    /// @return the letter of cell @param p
    char tile(const position & p) const
    {
        return lake_.tile(number(p));
    }

    /// @return the moves a session may make
    std::vector<action_type> actions(const state_type &) const
    {
        return {0, 1, 2, 3};
    }

    /// @brief start a new episode @return the start
    state_type reset()
    {
        return cell(lake_.reset());
    }

    /// @brief put the walker on cell @param p
    void place(const position & p)
    {
        lake_.place(number(p));
    }

    /// @brief try the move @param a (the ice decides) @return the cell reached, the reward and whether the game ended
    step_result<state_type> step(const action_type & a)
    {
        // the lake numbers its moves the other way round: left, down, right, up
        const auto out = lake_.step(3 - a);
        step_result<state_type> result;
        result.next       = cell(out.next);
        result.reward     = out.reward;
        result.terminated = out.terminated;
        return result;
    }

private:
    /// @return the lake's number of cell @param p
    unsigned int number(const position & p) const
    {
        return static_cast<unsigned int>(p.y * lake_.columns() + p.x);
    }

    /// @return the cell of the lake's number @param n
    position cell(unsigned int n) const
    {
        return position{static_cast<int>(n) % lake_.columns(), static_cast<int>(n) / lake_.columns()};
    }

    frozen_lake lake_;
};

/// @return the cell @param p kept on a board of @param columns by @param rows: the edge keeps a move in
inline position on_board(const position & p,
                         int columns,
                         int rows)
{
    return position{std::max(0, std::min(columns - 1, p.x)), std::max(0, std::min(rows - 1, p.y))};
}

/// @return the move @param tried turned by @param turn
inline unsigned int turned(unsigned int tried,
                           int turn)
{
    return (tried + static_cast<unsigned int>(turn + 4)) % 4;
}

/**
 * @brief which way the ice turned a move
 * @param from is the cell it left, @param tried the move chosen, @param to the cell reached
 * @param columns and @param rows are the size of the board
 * @return 0 if the move went as chosen, -1 if it turned to the left of it, 1 to the right
 */
inline int turn_of(const position & from,
                   unsigned int tried,
                   const position & to,
                   int columns,
                   int rows)
{
    for (int turn : {0, -1, 1}) {
        if (on_board(moved_to(from, turned(tried, turn)), columns, rows) == to) {
            return turn;
        }
    }
    return 0;
}

/**
 * @brief what to say about a step on the ice
 * @param names are the names of the moves
 * @param from is the cell it left, @param tried the move chosen, @param to the cell reached
 * @param columns and @param rows are the size of the board
 * @return a sentence if the ice got in the way, empty if the move went as chosen
 */
inline std::string slip_words(const std::vector<std::string> & names,
                              const position & from,
                              unsigned int tried,
                              const position & to,
                              int columns,
                              int rows)
{
    const int turn = turn_of(from, tried, to, columns, rows);
    if (turn == 0) {
        return "";
    }
    if (from == to) {
        return "It tried going " + names[tried] + ", but slid into the edge and stayed where it was.";
    }
    return "It tried going " + names[tried] + ", but the ice sent it " + names[turned(tried, turn)] + ".";
}

/**
 * @struct spot
 * @brief a place on the board, between cells
 */
struct spot
{
    double x;
    double y;
};

/**
 * @brief where the traveller is during a step: it sets off the way it chose, then the ice takes it
 * @param from is the cell it left, @param tried the move chosen, @param to the cell reached
 * @param turn says how the ice turned it, as `turn_of` does
 * @param t is how far the step has got, from 0 to 1
 * @return a straight line if it went as chosen, and otherwise a start the wrong way, then a skid to where it ended
 */
inline spot slide_path(const position & from,
                       unsigned int tried,
                       const position & to,
                       int turn,
                       double t)
{
    const double pivot = 0.3;   // how far it gets the way it chose
    const double turning = 0.4; // the part of the step before the ice takes over
    t = std::max(0.0, std::min(1.0, t));
    if (turn == 0) {
        return spot{from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t};
    }
    const position aim = moved_to(from, tried);
    const spot out{from.x + (aim.x - from.x) * pivot, from.y + (aim.y - from.y) * pivot};
    if (t < turning) {
        const double part = t / turning;
        return spot{from.x + (out.x - from.x) * part, from.y + (out.y - from.y) * part};
    }
    const double part = (t - turning) / (1 - turning);
    return spot{out.x + (to.x - out.x) * part, out.y + (to.y - out.y) * part};
}

/**
 * @class frozen_session
 * @brief a learner on the ice: a hole is a fall, and the curve is how often it has been winning lately
 * @tparam rule is the learner: `q_learning_rule`, `sarsa_rule` or `expected_sarsa_rule`
 */
template <class rule>
class frozen_session : public tabular_session<frozen_grid, rule>
{
public:
    using base = tabular_session<frozen_grid, rule>;

    /// the episodes the curve averages over
    static constexpr std::size_t window = 100;

    using base::base;

    void step() override
    {
        base::step();
        this->punished_ = this->game_.tile(this->s_) == 'H';
        if (!this->playing_) {
            record();
        }
    }

    void reset() override
    {
        base::reset();
        rates_.clear();
        wins_ = 0;
    }

    const std::vector<double> & curve() const override { return rates_; }
    double curve_now() const override { return rates_.empty() ? 0.0 : rates_.back(); }
    std::string curve_label() const override { return "share won, of the last 100 episodes"; }
    std::string curve_help() const override { return "Of its last 100 episodes, how many reached the treasure. Even the best plan loses some: the ice sees to that."; }
    std::string tally() const override { return "won " + std::to_string(wins_) + " of " + std::to_string(this->returns_.size()); }

private:
    /// @brief one more episode on the curve
    void record()
    {
        const auto & all = this->returns_;
        wins_ += all.back() > 0 ? 1u : 0u;
        const std::size_t count = std::min(window, all.size());
        rates_.push_back(std::accumulate(all.end() - static_cast<std::ptrdiff_t>(count), all.end(), 0.0) / static_cast<double>(count));
    }

    std::vector<double> rates_;
    std::size_t wins_ = 0;
};
} // namespace lessons
} // namespace games
#endif
