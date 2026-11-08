#ifndef RELEARN_GAMES_VIEWS_HPP
#define RELEARN_GAMES_VIEWS_HPP
/**
 * How each game is drawn, for `games::watched` (have a look at watch.hpp).
 *
 * - The grid games (Cliff Walking, Windy Gridworld, the Dyna maze, FrozenLake and the racetrack) are drawn cell by
 *   cell through a `grid_view`, so many pixels to a cell, and the agent as a square on its cell. Given the learnt
 *   values, each cell is coloured by its best value, from `colours::cold` (the lowest) to `colours::hot` (the
 *   highest), so the learning can be seen spreading out from the goal.
 * - Mountain Car is drawn as the hill, the car on it and the flag; given the values of a `discretised` game, the
 *   (position, velocity) plane is drawn next to it, each cell coloured by its best value: the book's Figure 10.1,
 *   with the colours the other way round, since the values are minus the cost-to-go.
 * - Cart-pole is drawn as the track, the cart and the pole, leaning as it does.
 */
#include "watch.hpp"
#include "cart_pole.hpp"
#include "cliff_walking.hpp"
#include "discretised.hpp"
#include "dyna_maze.hpp"
#include "frozen_lake.hpp"
#include "mountain_car.hpp"
#include "racetrack.hpp"
#include "windy_gridworld.hpp"
#include <cmath>
#include <limits>

namespace games
{
/**
 * @class heat
 * @brief the colour of each state by its best learnt value, from `colours::cold` to `colours::hot`
 * @tparam game_type is the game whose states are coloured
 *
 * The lowest and highest values are found over the states given, so the colours always use the whole range;
 * a state without a value yet (none of its actions tried) keeps the colour it would have had anyway.
 */
template <class game_type>
class heat
{
public:
    using state_type  = typename game_type::state_type;
    using policy_type = typename types<game_type>::policy;

    /// @brief the range of the values of @param states in @param memory (which may be null: then there is no heat)
    template <class container>
    heat(const policy_type * memory,
         const container & states)
    : memory_(memory)
    {
        if (!memory_) {
            return;
        }
        for (const auto & s : states) {
            const double v = value(s);
            if (!std::isnan(v)) {
                lowest_  = std::min(lowest_, v);
                highest_ = std::max(highest_, v);
            }
        }
    }

    /// @return true if some state has a value to show
    bool any() const
    {
        return memory_ && lowest_ <= highest_;
    }

    /// @return the colour of @param s: by its best value, or @param otherwise if it has none
    rgb of(const state_type & s,
           const rgb & otherwise) const
    {
        if (!any()) {
            return otherwise;
        }
        const double v = value(s);
        if (std::isnan(v)) {
            return otherwise;
        }
        const double span = highest_ - lowest_;
        return gradient(span > 0 ? (v - lowest_) / span : 1.0, colours::cold, colours::hot);
    }

private:
    /// @return the best value of @param s, NaN if none of its actions has one
    double value(const state_type & s) const
    {
        return memory_->best_value(typename types<game_type>::state(s));
    }

    // the learnt values, or null
    const policy_type * memory_;
    // the range of the values found
    double lowest_  = std::numeric_limits<double>::infinity();
    double highest_ = -std::numeric_limits<double>::infinity();
};

/// @return every cell of a grid of @param columns x @param rows, row by row, for `heat`
inline std::vector<position> positions(int columns,
                                       int rows)
{
    std::vector<position> all;
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < columns; x++) {
            all.push_back({x, y});
        }
    }
    return all;
}

/// @return the states 0 to @param count - 1, for `heat`
inline std::vector<unsigned int> numbered(unsigned int count)
{
    std::vector<unsigned int> all;
    for (unsigned int s = 0; s < count; s++) {
        all.push_back(s);
    }
    return all;
}

/**
 * @class grid_view
 * @brief what the grid games share: so many pixels to a cell, the cells drawn one by one, and the agent as a
 *        square on its cell
 */
class grid_view
{
public:
    /// pixels to a cell, unless the view says otherwise
    static constexpr unsigned int scale = 4;

    /// @brief a grid drawn @param pixels_per_cell pixels to a cell
    explicit grid_view(unsigned int pixels_per_cell = scale)
    : cell_(static_cast<int>(pixels_per_cell))
    {}

    /// @return the pixels to a cell
    int cell() const
    {
        return cell_;
    }

    /// @return the width of a grid of @param columns cells, in pixels
    unsigned int width_of(int columns) const
    {
        return static_cast<unsigned int>(columns * cell_);
    }

    /// @return the height of a grid of @param rows cells, in pixels
    unsigned int height_of(int rows) const
    {
        return static_cast<unsigned int>(rows * cell_);
    }

    /**
     * @brief draw every cell of a grid of @param columns x @param rows
     * @param colour is called as `colour(x, y)` and gives the colour of the cell in column x and row y
     */
    template <class colouring>
    void cells(canvas & c,
               int columns,
               int rows,
               const colouring & colour) const
    {
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < columns; x++) {
                c.fill(x * cell_, y * cell_, static_cast<unsigned int>(cell_), static_cast<unsigned int>(cell_),
                       colour(x, y));
            }
        }
    }

    /// @brief draw the agent on the cell in column @param x and row @param y: a square inside the cell, one pixel
    ///        in from its edges when the cell is big enough (more than 2 pixels), the whole cell otherwise
    void agent(canvas & c,
               int x,
               int y,
               const rgb & colour = colours::agent) const
    {
        const int inset = cell_ > 2 ? 1 : 0;
        c.fill(x * cell_ + inset, y * cell_ + inset, static_cast<unsigned int>(cell_ - 2 * inset),
               static_cast<unsigned int>(cell_ - 2 * inset), colour);
    }

private:
    // the pixels to a cell
    int cell_;
};

/**
 * @brief the view of Cliff Walking: the cliff in `colours::danger`, the start and the goal, and the agent
 */
template <>
struct view<cliff_walking>
{
    using game_type = cliff_walking;

    static unsigned int width()
    {
        return grid_view().width_of(game_type::width);
    }

    static unsigned int height()
    {
        return grid_view().height_of(game_type::height);
    }

    static void draw(canvas & c,
                     const game_type &,
                     const position & at,
                     const types<game_type>::policy * memory)
    {
        const grid_view grid;
        heat<game_type> values(memory, positions(game_type::width, game_type::height));
        grid.cells(c, game_type::width, game_type::height, [&](int x, int y) {
            const position p{x, y};
            if (game_type::cliff(p)) {
                return colours::danger;
            }
            if (p == game_type::goal()) {
                return colours::goal;
            }
            if (p == game_type::start()) {
                return values.of(p, colours::start);
            }
            return values.of(p, colours::floor);
        });
        grid.agent(c, at.x, at.y);
    }
};

/**
 * @brief the view of Windy Gridworld: the columns shaded by their wind, the start, the goal, and the agent
 */
template <>
struct view<windy_gridworld>
{
    using game_type = windy_gridworld;

    static unsigned int width()
    {
        return grid_view().width_of(game_type::width);
    }

    static unsigned int height()
    {
        return grid_view().height_of(game_type::height);
    }

    static void draw(canvas & c,
                     const game_type &,
                     const position & at,
                     const types<game_type>::policy * memory)
    {
        const grid_view grid;
        heat<game_type> values(memory, positions(game_type::width, game_type::height));
        grid.cells(c, game_type::width, game_type::height, [&](int x, int y) {
            const position p{x, y};
            if (p == game_type::goal()) {
                return colours::goal;
            }
            if (p == game_type::start()) {
                return values.of(p, colours::start);
            }
            // the stronger the wind (0 to 2), the lighter the column
            const rgb windy = gradient(game_type::wind(x) / 2.0, colours::floor, colours::wall);
            return values.of(p, windy);
        });
        grid.agent(c, at.x, at.y);
    }
};

/**
 * @brief the view of the Dyna maze: the walls, the start, the goal, and the agent
 */
template <>
struct view<dyna_maze>
{
    using game_type = dyna_maze;

    /// the picture is that of a maze at scale 1; a maze scaled by 2 or 4 is drawn at the same size, with smaller
    /// cells (another scale gets as many pixels to a cell as fit, and does not fill the picture)
    static unsigned int width()
    {
        return grid_view().width_of(game_type::width);
    }

    static unsigned int height()
    {
        return grid_view().height_of(game_type::height);
    }

    static void draw(canvas & c,
                     const game_type & maze,
                     const position & at,
                     const types<game_type>::policy * memory)
    {
        const int columns = maze.columns();
        const int rows    = maze.rows();
        const grid_view grid(width() / static_cast<unsigned int>(columns));
        heat<game_type> values(memory, positions(columns, rows));
        grid.cells(c, columns, rows, [&](int x, int y) {
            const position p{x, y};
            if (maze.wall(p)) {
                return colours::wall;
            }
            if (p == maze.goal()) {
                return colours::goal;
            }
            if (p == maze.start()) {
                return values.of(p, colours::start);
            }
            return values.of(p, colours::floor);
        });
        grid.agent(c, at.x, at.y);
    }
};

/**
 * @brief the view of FrozenLake: the ice, the holes, the start, the goal, and the agent
 */
template <>
struct view<frozen_lake>
{
    using game_type = frozen_lake;

    /// the picture is sized for the 8 x 8 lake; the 4 x 4 one is drawn at the same size, with bigger cells
    static unsigned int width()
    {
        return grid_view().width_of(8);
    }

    static unsigned int height()
    {
        return grid_view().height_of(8);
    }

    static void draw(canvas & c,
                     const game_type & lake,
                     const unsigned int & at,
                     const types<game_type>::policy * memory)
    {
        const int columns = lake.columns();
        const int rows    = lake.rows();
        const grid_view grid(width() / static_cast<unsigned int>(columns));
        heat<game_type> values(memory, numbered(lake.states()));
        grid.cells(c, columns, rows, [&](int x, int y) {
            const unsigned int s = static_cast<unsigned int>(y * columns + x);
            switch (lake.tile(s)) {
                case 'H':
                    return colours::danger;
                case 'G':
                    return colours::goal;
                case 'S':
                    return values.of(s, colours::start);
                default:
                    return values.of(s, colours::ice);
            }
        });
        grid.agent(c, static_cast<int>(at) % columns, static_cast<int>(at) / columns);
    }
};

/**
 * @brief the view of the racetrack: the track, the start and finish lines, and the car with its velocity as a tail;
 *        the finish line lights up when the car crosses it
 */
template <>
struct view<racetrack>
{
    using game_type = racetrack;
    /// pixels to a cell: the track is tall, so its cells are small
    static constexpr unsigned int scale = 2;

    /// the picture is sized for the default track; another track is drawn at its own size, pixels outside are lost
    static unsigned int width()
    {
        return grid_view(scale).width_of(static_cast<int>(game_type::default_track().front().size()));
    }

    static unsigned int height()
    {
        return grid_view(scale).height_of(static_cast<int>(game_type::default_track().size()));
    }

    static void draw(canvas & c,
                     const game_type & track,
                     const unsigned int & at,
                     const types<game_type>::policy * memory)
    {
        const grid_view grid(scale);
        const int s = grid.cell();
        const bool finished = at == track.finished();
        // the heat is that of the cells at rest: one value per cell, of the 25 velocities
        std::vector<unsigned int> at_rest;
        for (int r = 0; r < track.rows(); r++) {
            for (int col = 0; col < track.columns(); col++) {
                if (track.tile(r, col) != '#') {
                    at_rest.push_back(track.state_of(r, col, 0, 0));
                }
            }
        }
        heat<game_type> values(memory, at_rest);
        grid.cells(c, track.columns(), track.rows(), [&](int col, int r) {
            switch (track.tile(r, col)) {
                case '#':
                    return colours::wall;
                case 'F':
                    // the finish line lights up once the car has crossed it
                    return finished ? colours::agent : colours::goal;
                case 'S':
                    return values.of(track.state_of(r, col, 0, 0), colours::start);
                default:
                    return values.of(track.state_of(r, col, 0, 0), colours::floor);
            }
        });
        if (finished) {
            return;
        }
        // the car, and its velocity as a tail behind it: vx is to the right and vy upwards
        const int r   = track.row(at);
        const int col = track.column(at);
        c.line(col * s, r * s + s - 1, (col - track.vx(at)) * s, (r + track.vy(at)) * s + s - 1, colours::danger);
        grid.agent(c, col, r);
    }
};

/**
 * @brief the view of Mountain Car: the hill, the car and the flag
 */
template <>
struct view<mountain_car>
{
    using game_type = mountain_car;

    static unsigned int width()
    {
        return 64;
    }

    static unsigned int height()
    {
        return 32;
    }

    /// @return the pixel column of position @param p
    static int column(double p)
    {
        const double t = (p - game_type::min_position) / (game_type::max_position - game_type::min_position);
        return static_cast<int>(t * (width() - 1) + 0.5);
    }

    /// @return the pixel row of the hill at position @param p: the book's height is sin(3p), drawn between rows 10
    ///         (the hilltops) and 30 (the bottom of the valley), leaving room for the flag above
    static int row(double p)
    {
        const double h = std::sin(3 * p);
        return static_cast<int>((1 - h) / 2 * (height() - 12) + 10 + 0.5);
    }

    static void draw(canvas & c,
                     const game_type &,
                     const std::array<double, 2> & at,
                     const types<game_type>::policy *)
    {
        c.fill(0, 0, width(), height(), colours::sky);
        // the hill, filled down to the ground
        for (int x = 0; x < static_cast<int>(width()); x++) {
            const double p = game_type::min_position + x * (game_type::max_position - game_type::min_position) / (width() - 1);
            const int top = row(p);
            c.line(x, top, x, static_cast<int>(height()) - 1, colours::ground);
        }
        // the flag at the goal
        const int fx = column(game_type::goal_position);
        const int fy = row(game_type::goal_position);
        c.line(fx, fy - 6, fx, fy - 1, colours::white);
        c.fill(fx + 1, fy - 6, 3, 2, colours::goal);
        // the car, sitting on the hill
        const int cx = column(at[0]);
        const int cy = row(at[0]);
        c.fill(cx - 1, cy - 2, 3, 2, colours::agent);
    }
};

/**
 * @brief the view of Mountain Car cut into a grid: the hill, and the (position, velocity) plane coloured by the
 *        best value of each cell (the book's Figure 10.1, the cost-to-go, with the colours the other way round:
 *        hot where the values are highest)
 */
template <>
struct view<discretised<mountain_car>>
{
    using game_type = discretised<mountain_car>;
    using scene     = view<mountain_car>;

    /// the hill on the left, the plane on the right, a pixel apart
    static unsigned int width()
    {
        return 2 * scene::width() + 1;
    }

    static unsigned int height()
    {
        return scene::height();
    }

    /// @return the pixel column of the plane holding position @param p, from 0: the plane's columns cut the
    ///         positions into `scene::width()` equal parts
    static int plane_column(const game_type & game,
                            double p)
    {
        const auto & bin = game.grid()[0];
        const int x = static_cast<int>((p - bin.low) / (bin.high - bin.low) * scene::width());
        return std::max(0, std::min(static_cast<int>(scene::width()) - 1, x));
    }

    /// @return the pixel row of the plane holding velocity @param v: the highest velocity at the top
    static int plane_row(const game_type & game,
                         double v)
    {
        const auto & bin = game.grid()[1];
        const int y = static_cast<int>((bin.high - v) / (bin.high - bin.low) * height());
        return std::max(0, std::min(static_cast<int>(height()) - 1, y));
    }

    static void draw(canvas & c,
                     const game_type & game,
                     const unsigned int &,
                     const types<game_type>::policy * memory)
    {
        scene::draw(c, game.inner(), game.raw(), nullptr);
        // the plane: positions along x, velocities along y (the highest at the top), each pixel the cell of its
        // centre
        const int left = static_cast<int>(scene::width()) + 1;
        heat<game_type> values(memory, numbered(game.cells()));
        const auto & grid = game.grid();
        for (int y = 0; y < static_cast<int>(height()); y++) {
            for (int x = 0; x < static_cast<int>(scene::width()); x++) {
                const double p = grid[0].low + (x + 0.5) / scene::width() * (grid[0].high - grid[0].low);
                const double v = grid[1].high - (y + 0.5) / height() * (grid[1].high - grid[1].low);
                c.plot(left + x, y, values.of(game.cell({p, v}), colours::floor));
            }
        }
        // the car, in the plane: a 3 x 3 square round the pixel its position and velocity fall in
        const int px = left + plane_column(game, game.raw()[0]);
        const int py = plane_row(game, game.raw()[1]);
        c.fill(px - 1, py - 1, 3, 3, colours::agent);
    }
};

/**
 * @brief the view of cart-pole: the track, the cart and the pole
 */
template <>
struct view<cart_pole>
{
    using game_type = cart_pole;

    static unsigned int width()
    {
        return 64;
    }

    static unsigned int height()
    {
        return 24;
    }

    static void draw(canvas & c,
                     const game_type &,
                     const std::array<double, 4> & at,
                     const types<game_type>::policy *)
    {
        c.fill(0, 0, width(), height(), colours::sky);
        const int ground = static_cast<int>(height()) - 3;
        c.line(0, ground + 1, static_cast<int>(width()) - 1, ground + 1, colours::ground);
        // the cart, along the track from -2.4 to 2.4
        const double t  = (at[0] + game_type::position_limit) / (2 * game_type::position_limit);
        const int cx    = static_cast<int>(t * (width() - 1) + 0.5);
        c.fill(cx - 3, ground - 2, 7, 3, colours::wall);
        // the pole, leaning by its angle (positive to the right), 14 pixels long
        const int length = 14;
        const int tip_x = cx + static_cast<int>(length * std::sin(at[2]) + 0.5);
        const int tip_y = ground - 2 - static_cast<int>(length * std::cos(at[2]) + 0.5);
        c.line(cx, ground - 2, tip_x, tip_y, colours::agent);
    }
};

/**
 * @brief the view of any other game cut into a grid: that of the game it wraps, drawn from its continuous state
 */
template <class game_type>
struct view<discretised<game_type>>
{
    using scene = view<game_type>;

    static unsigned int width()
    {
        return scene::width();
    }

    static unsigned int height()
    {
        return scene::height();
    }

    static void draw(canvas & c,
                     const discretised<game_type> & game,
                     const unsigned int &,
                     const typename types<discretised<game_type>>::policy *)
    {
        scene::draw(c, game.inner(), game.raw(), nullptr);
    }
};
} // namespace games
#endif
