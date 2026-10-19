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
/**
 * @struct bins
 * @brief equal-width bins of one state variable, between `low` and `high`
 */
struct bins
{
    /// the lowest value: anything below falls in the first bin
    double low;
    /// the highest value: anything above falls in the last bin
    double high;
    /// the number of bins
    unsigned int count;

    /// @return the bin of @param value, from 0 to `count - 1`
    unsigned int of(double value) const
    {
        // values outside the range fall in the first or the last bin
        if (value <= low) {
            return 0;
        }
        if (value >= high) {
            return count - 1;
        }
        // values inside the range: the bin they fall in, counting from `low`
        auto b = static_cast<unsigned int>((value - low) / (high - low) * count);
        return b < count ? b : count - 1;
    }
};

/**
 * @class discretised
 * @brief wraps a game with a continuous state, so that a tabular learner can play it:
 *        the learner sees the number of the cell the continuous state falls in
 */
template <class game_type>
class discretised
{
public:
    /// a state is the number of a cell
    using state_type  = unsigned int;
    /// the actions are those of the wrapped game
    using action_type = typename game_type::action_type;

    /**
     * @brief wrap @param game
     * @param grid gives the bins of each state variable, in the order of the state
     */
    discretised(game_type game,
                std::vector<bins> grid)
    : game_(std::move(game)),
      grid_(std::move(grid))
    {}

    /// @return the wrapped game, e.g. for its step limit
    game_type & inner()
    {
        return game_;
    }

    /// @return the continuous state of the last step
    const typename game_type::state_type & raw() const
    {
        return raw_;
    }

    /// @return the number of cells: the product of the bins of every variable
    unsigned int cells() const
    {
        unsigned int n = 1;
        for (const auto & b : grid_) {
            n *= b.count;
        }
        return n;
    }

    /// @return the cell of continuous state @param s
    state_type cell(const typename game_type::state_type & s) const
    {
        // number the cells like digits of a number, one digit per state variable
        unsigned int c = 0;
        for (std::size_t i = 0; i < grid_.size(); i++) {
            c = c * grid_[i].count + grid_[i].of(s[i]);
        }
        return c;
    }

    /// @brief start a new episode @return the cell of the starting state
    state_type reset()
    {
        raw_ = game_.reset();
        return cell(raw_);
    }

    /// @brief take action @param a in the wrapped game @return the cell reached, the reward and whether the game ended
    step_result<state_type> step(const action_type & a)
    {
        // play the step in the continuous game, and remember the state reached
        auto out = game_.step(a);
        raw_ = out.next;
        // the same outcome, with the state replaced by its cell
        step_result<state_type> result;
        result.next       = cell(out.next);
        result.reward     = out.reward;
        result.terminated = out.terminated;
        return result;
    }

    /// @return the legal actions of the wrapped game in its current state
    std::vector<action_type> actions(const state_type &) const
    {
        return game_.actions(raw_);
    }

private:
    // the wrapped game
    game_type game_;
    // the bins of each state variable
    std::vector<bins> grid_;
    // the continuous state of the last step
    typename game_type::state_type raw_{};
};
} // namespace games
#endif
