#ifndef RELEARN_GAMES_DISCRETISED_HPP
#define RELEARN_GAMES_DISCRETISED_HPP
/**
 * Play a game with a continuous state (cart-pole, mountain car) with a tabular
 * learner, by cutting each state variable into bins.
 *
 *     games::discretised<games::cart_pole> game(games::cart_pole(seed),
 *                                               {{-2.4, 2.4, 3}, {-3, 3, 3}, ...});
 *
 * Each variable gets `bins` equal intervals between `low` and `high`; values
 * outside fall in the first or last bin. The state seen by the learner is the
 * cell number, the same for all states inside one cell.
 */
#include "game.hpp"
#include <cmath>
#include <utility>
#include <vector>

namespace games
{
/// equal-width bins of one state variable
struct bins
{
    double low;
    double high;
    unsigned int count;

    unsigned int of(double value) const
    {
        if (value <= low) return 0;
        if (value >= high) return count - 1;
        auto b = static_cast<unsigned int>((value - low) / (high - low) * count);
        return b < count ? b : count - 1;
    }
};

template <class game_type>
class discretised
{
public:
    using state_type  = unsigned int;                 // cell number
    using action_type = typename game_type::action_type;

    discretised(game_type game, std::vector<bins> grid)
    : game_(std::move(game)), grid_(std::move(grid))
    {}

    /// the wrapped game, e.g. for its step limit
    game_type & inner() { return game_; }
    /// the continuous state of the last step
    const typename game_type::state_type & raw() const { return raw_; }
    /// the number of cells
    unsigned int cells() const
    {
        unsigned int n = 1;
        for (const auto & b : grid_) n *= b.count;
        return n;
    }

    /// the cell of continuous state @param s
    state_type cell(const typename game_type::state_type & s) const
    {
        unsigned int c = 0;
        for (std::size_t i = 0; i < grid_.size(); i++) {
            c = c * grid_[i].count + grid_[i].of(s[i]);
        }
        return c;
    }

    state_type reset()
    {
        raw_ = game_.reset();
        return cell(raw_);
    }

    step_result<state_type> step(const action_type & a)
    {
        auto out = game_.step(a);
        raw_ = out.next;
        step_result<state_type> result;
        result.next = cell(out.next);
        result.reward = out.reward;
        result.terminated = out.terminated;
        return result;
    }

    std::vector<action_type> actions(const state_type &) const
    {
        return game_.actions(raw_);
    }

private:
    game_type game_;
    std::vector<bins> grid_;
    typename game_type::state_type raw_{};
};
} // namespace games
#endif
