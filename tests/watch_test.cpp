/**
 * Tests of the watch mode: the colours and how they are written for the terminal, the canvas and what it
 * renders, worked out by hand; the views of the games, pixel by pixel; and that a watched game plays exactly as
 * the game it wraps, and draws when, and only when, it should.
 */
#include "../examples/games/views.hpp"
#include <sstream>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

namespace
{
/// @return how many times @param piece occurs in @param text
std::size_t count(const std::string & text,
                  const std::string & piece)
{
    std::size_t n = 0;
    for (std::size_t at = text.find(piece); at != std::string::npos; at = text.find(piece, at + piece.size())) {
        n++;
    }
    return n;
}

/// @brief a random agent, the same on a game and on the watched game, so that their episodes can be compared
template <class game_type>
struct random_agent
{
    std::mt19937 prng;

    typename game_type::action_type operator()(const typename game_type::state_type &,
                                               const std::vector<typename game_type::action_type> & legal)
    {
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        return legal[pick(prng)];
    }
};
} // namespace

TEST_CASE("colours: equality and the gradient", "[watch][colours]")
{
    REQUIRE(games::rgb{1, 2, 3} == games::rgb{1, 2, 3});
    REQUIRE(games::rgb{1, 2, 3} != games::rgb{1, 2, 4});
    // half way from black to white is 128 (255 / 2 rounded up)
    REQUIRE(games::gradient(0.5, games::colours::black, games::colours::white) == games::rgb{128, 128, 128});
    REQUIRE(games::gradient(0.0, games::colours::cold, games::colours::hot) == games::colours::cold);
    REQUIRE(games::gradient(1.0, games::colours::cold, games::colours::hot) == games::colours::hot);
    // t is clamped
    REQUIRE(games::gradient(-3.0, games::colours::cold, games::colours::hot) == games::colours::cold);
    REQUIRE(games::gradient(7.0, games::colours::cold, games::colours::hot) == games::colours::hot);
    // a quarter of the way from 0 to 100 is 25
    REQUIRE(games::gradient(0.25, games::rgb{0, 0, 0}, games::rgb{100, 100, 100}) == games::rgb{25, 25, 25});
}

TEST_CASE("truecolour: the escape sequences, and whether the terminal has it", "[watch][palette]")
{
    REQUIRE(games::truecolour::foreground({1, 2, 3}) == "\x1b[38;2;1;2;3m");
    REQUIRE(games::truecolour::background({255, 0, 128}) == "\x1b[48;2;255;0;128m");
    // COLORTERM says truecolor or 24bit, or it does not, or it is not set at all
    REQUIRE(games::truecolour::available("truecolor"));
    REQUIRE(games::truecolour::available("24bit"));
    REQUIRE(!games::truecolour::available("xterm-256color"));
    REQUIRE(!games::truecolour::available(""));
    REQUIRE(!games::truecolour::available(nullptr));
    setenv("COLORTERM", "truecolor", 1);
    REQUIRE(games::truecolour::available());
    setenv("COLORTERM", "yes", 1);
    REQUIRE(!games::truecolour::available());
    unsetenv("COLORTERM");
    REQUIRE(!games::truecolour::available());
}

TEST_CASE("colour_256: the nearest of the 6 x 6 x 6 colours", "[watch][palette]")
{
    // the cube's values are 0, 95, 135, 175, 215 and 255: each component goes to the nearest
    REQUIRE(games::colour_256::level(0) == 0);
    REQUIRE(games::colour_256::level(95) == 1);
    REQUIRE(games::colour_256::level(135) == 2);
    REQUIRE(games::colour_256::level(175) == 3);
    REQUIRE(games::colour_256::level(215) == 4);
    REQUIRE(games::colour_256::level(255) == 5);
    // the first boundary is half way from 0 to 95: 47 is still level 0, 48 is level 1
    REQUIRE(games::colour_256::level(47) == 0);
    REQUIRE(games::colour_256::level(48) == 1);
    // half way from 95 to 135 is 115, as near to both, and goes to the lower: 114 and 115 are level 1, 116 is level 2
    REQUIRE(games::colour_256::level(114) == 1);
    REQUIRE(games::colour_256::level(115) == 1);
    REQUIRE(games::colour_256::level(116) == 2);
    REQUIRE(games::colour_256::level(254) == 5);
    REQUIRE(games::colour_256::level(1) == 0);
    // black is 16, white 231, red 196, blue 21: 16 + 36 r + 6 g + b, each a level from 0 to 5
    REQUIRE(games::colour_256::index({0, 0, 0}) == 16);
    REQUIRE(games::colour_256::index({255, 255, 255}) == 231);
    REQUIRE(games::colour_256::index({255, 0, 0}) == 196);
    REQUIRE(games::colour_256::index({0, 0, 255}) == 21);
    // 128 and 127 are nearest to 135, level 2: 16 + 72 + 12 + 2, and 16 + 72
    REQUIRE(games::colour_256::index({128, 128, 128}) == 102);
    REQUIRE(games::colour_256::index({127, 0, 0}) == 88);
    // the dark floor (40, 40, 48) is nearly black, with a touch of blue, and the sky (20, 24, 40) is black
    REQUIRE(games::colour_256::index(games::colours::floor) == 17);
    REQUIRE(games::colour_256::index(games::colours::sky) == 16);
    REQUIRE(games::colour_256::foreground({0, 0, 255}) == "\x1b[38;5;21m");
    REQUIRE(games::colour_256::background({255, 0, 0}) == "\x1b[48;5;196m");
}

TEST_CASE("canvas: pixels, outside ignored, fill and clear", "[watch][canvas]")
{
    games::canvas c(3, 2, games::colours::grey);
    REQUIRE(c.width() == 3);
    REQUIRE(c.height() == 2);
    REQUIRE(c.rows() == 1);
    REQUIRE(c.at(0, 0) == games::colours::grey);
    c.plot(2, 1, games::colours::agent);
    REQUIRE(c.at(2, 1) == games::colours::agent);
    REQUIRE(c.at(2, 0) == games::colours::grey);
    // outside: nothing happens, and reading gives black
    c.plot(3, 0, games::colours::agent);
    c.plot(-1, 0, games::colours::agent);
    c.plot(0, 2, games::colours::agent);
    REQUIRE(c.at(3, 0) == games::colours::black);
    REQUIRE(c.at(-1, 0) == games::colours::black);
    REQUIRE(c.at(0, 2) == games::colours::black);
    // a rectangle, partly outside
    c.fill(1, 0, 5, 1, games::colours::goal);
    REQUIRE(c.at(1, 0) == games::colours::goal);
    REQUIRE(c.at(2, 0) == games::colours::goal);
    REQUIRE(c.at(0, 0) == games::colours::grey);
    REQUIRE(c.at(1, 1) == games::colours::grey);
    c.clear(games::colours::black);
    REQUIRE(c.at(2, 1) == games::colours::black);
    // an odd height has a half-filled last row
    REQUIRE(games::canvas(2, 3).rows() == 2);
    REQUIRE(games::canvas(2, 1).rows() == 1);
}

TEST_CASE("canvas: lines", "[watch][canvas]")
{
    games::canvas c(5, 5);
    // a diagonal
    c.line(0, 0, 3, 3, games::colours::white);
    for (int i = 0; i < 4; i++) {
        REQUIRE(c.at(i, i) == games::colours::white);
    }
    REQUIRE(c.at(1, 0) == games::colours::black);
    REQUIRE(c.at(4, 4) == games::colours::black);
    // a shallow line: 5 pixels, one per column, both ends included, never going back down
    games::canvas d(5, 3);
    d.line(0, 0, 4, 2, games::colours::white);
    std::size_t lit = 0;
    int last_y = -1;
    for (int x = 0; x < 5; x++) {
        int found = -1;
        for (int y = 0; y < 3; y++) {
            if (d.at(x, y) == games::colours::white) {
                lit++;
                found = y;
            }
        }
        REQUIRE(found >= last_y);
        last_y = found;
    }
    REQUIRE(lit == 5);
    REQUIRE(d.at(0, 0) == games::colours::white);
    REQUIRE(d.at(4, 2) == games::colours::white);
    // a line drawn backwards lights the same pixels
    games::canvas e(5, 3);
    e.line(4, 2, 0, 0, games::colours::white);
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 3; y++) {
            REQUIRE(e.at(x, y) == d.at(x, y));
        }
    }
    // a point
    games::canvas f(3, 3);
    f.line(1, 1, 1, 1, games::colours::white);
    REQUIRE(f.at(1, 1) == games::colours::white);
    REQUIRE(f.at(0, 1) == games::colours::black);
    REQUIRE(f.at(1, 0) == games::colours::black);
}

TEST_CASE("canvas: rendering two pixels per character", "[watch][canvas]")
{
    // one column, two rows: the top pixel is the text colour, the bottom one the background, under an upper half block
    games::canvas c(1, 2);
    c.plot(0, 0, {255, 0, 0});
    c.plot(0, 1, {0, 0, 255});
    REQUIRE(c.render<games::truecolour>() == "\x1b[38;2;255;0;0m\x1b[48;2;0;0;255m\xE2\x96\x80\x1b[0m\n");
    REQUIRE(c.render<games::colour_256>() == "\x1b[38;5;196m\x1b[48;5;21m\xE2\x96\x80\x1b[0m\n");
    // two columns, one row: the missing bottom row is black
    games::canvas d(2, 1);
    d.plot(0, 0, {1, 2, 3});
    d.plot(1, 0, {4, 5, 6});
    REQUIRE(d.render<games::truecolour>() == "\x1b[38;2;1;2;3m\x1b[48;2;0;0;0m\xE2\x96\x80"
                                             "\x1b[38;2;4;5;6m\x1b[48;2;0;0;0m\xE2\x96\x80\x1b[0m\n");
    // three rows make two lines
    games::canvas e(1, 3);
    REQUIRE(count(e.render<games::truecolour>(), "\n") == 2);
    REQUIRE(count(e.render<games::truecolour>(), "\xE2\x96\x80") == 2);
    REQUIRE(count(e.render<games::truecolour>(), "\x1b[0m") == 2);
}

TEST_CASE("learning curve: the frame, and the points scaled from the lowest to the highest", "[watch][curve]")
{
    games::canvas c(3, 4);
    games::plot_curve(c, 0, 0, 3, 4, {0.0, 10.0});
    // the axes: the left column and the bottom row
    REQUIRE(c.at(0, 0) == games::colours::axis);
    REQUIRE(c.at(0, 3) == games::colours::axis);
    REQUIRE(c.at(2, 3) == games::colours::axis);
    // the first value, the lowest, sits just above the axis; the second, the highest, at the top
    REQUIRE(c.at(1, 2) == games::colours::curve);
    REQUIRE(c.at(2, 0) == games::colours::curve);
    // the line between them passes through (1, 1) or (2, 1): one of them is lit, and the rest is background
    REQUIRE((c.at(1, 1) == games::colours::curve || c.at(2, 1) == games::colours::curve));
    REQUIRE(c.at(1, 0) == games::colours::sky);
    // no values: only the frame
    games::canvas d(3, 4);
    games::plot_curve(d, 0, 0, 3, 4, {});
    REQUIRE(d.at(1, 1) == games::colours::sky);
    REQUIRE(d.at(0, 0) == games::colours::axis);
    // equal values are drawn along the middle: with h = 4, row 1
    games::canvas e(4, 4);
    games::plot_curve(e, 0, 0, 4, 4, {5.0, 5.0, 5.0});
    REQUIRE(e.at(1, 1) == games::colours::curve);
    REQUIRE(e.at(2, 1) == games::colours::curve);
    REQUIRE(e.at(3, 1) == games::colours::curve);
    REQUIRE(e.at(1, 2) == games::colours::sky);
    // more values than columns are averaged into the columns: 4 values, 2 columns, the first two average to 0 and
    // the last two to 10
    games::canvas f(3, 4);
    games::plot_curve(f, 0, 0, 3, 4, {-5.0, 5.0, 8.0, 12.0});
    REQUIRE(f.at(1, 2) == games::colours::curve);
    REQUIRE(f.at(2, 0) == games::colours::curve);
    // averaged, not summed: 5 equal values into 3 columns (2, 2 and 1 of them) are a flat line along the middle,
    // whereas their sums (2, 2, 1) would not be
    games::canvas h(4, 4);
    games::plot_curve(h, 0, 0, 4, 4, {1.0, 1.0, 1.0, 1.0, 1.0});
    REQUIRE(h.at(1, 1) == games::colours::curve);
    REQUIRE(h.at(2, 1) == games::colours::curve);
    REQUIRE(h.at(3, 1) == games::colours::curve);
    REQUIRE(h.at(3, 2) == games::colours::sky);
    REQUIRE(h.at(1, 0) == games::colours::sky);
    // the oldest values go in the first column: 0, 10, 0, 10 into 2 columns are 5 and 5, a flat line, whereas dealt
    // round the columns they would be 0 and 10
    games::canvas k(3, 4);
    games::plot_curve(k, 0, 0, 3, 4, {0.0, 10.0, 0.0, 10.0});
    REQUIRE(k.at(1, 1) == games::colours::curve);
    REQUIRE(k.at(2, 1) == games::colours::curve);
    REQUIRE(k.at(2, 0) == games::colours::sky);
    REQUIRE(k.at(1, 2) == games::colours::sky);
    // drawn with an offset
    games::canvas g(6, 6);
    games::plot_curve(g, 2, 1, 3, 4, {0.0, 10.0});
    REQUIRE(g.at(2, 1) == games::colours::axis);
    REQUIRE(g.at(4, 1) == games::colours::curve);
    REQUIRE(g.at(0, 0) == games::colours::black);
}

TEST_CASE("heat: states coloured from the lowest to the highest value", "[watch][heat]")
{
    using game_type = games::cliff_walking;
    using types     = games::types<game_type>;
    types::policy memory;
    const games::position a{0, 0};
    const games::position b{1, 0};
    const games::position c{2, 0};
    memory.update(types::state(a), types::action(games::up), -10);
    memory.update(types::state(a), types::action(games::right), -20);
    memory.update(types::state(b), types::action(games::up), -40);
    std::vector<games::position> all = {a, b, c};
    games::heat<game_type> values(&memory, all);
    REQUIRE(values.any());
    // a's best value, -10, is the highest, b's, -40, the lowest, and c has none
    REQUIRE(values.of(a, games::colours::floor) == games::colours::hot);
    REQUIRE(values.of(b, games::colours::floor) == games::colours::cold);
    REQUIRE(values.of(c, games::colours::floor) == games::colours::floor);
    // half way between -40 and -10 is -25
    memory.update(types::state(c), types::action(games::up), -25);
    games::heat<game_type> three(&memory, all);
    REQUIRE(three.of(c, games::colours::floor) == games::gradient(0.5, games::colours::cold, games::colours::hot));
    // one value only is hot
    types::policy one;
    one.update(types::state(a), types::action(games::up), -3);
    REQUIRE(games::heat<game_type>(&one, all).of(a, games::colours::floor) == games::colours::hot);
    // no values at all, or no policy: the colour given
    types::policy empty;
    REQUIRE(!games::heat<game_type>(&empty, all).any());
    REQUIRE(games::heat<game_type>(&empty, all).of(a, games::colours::floor) == games::colours::floor);
    REQUIRE(!games::heat<game_type>(nullptr, all).any());
    REQUIRE(games::heat<game_type>(nullptr, all).of(a, games::colours::ice) == games::colours::ice);
    // only the states given set the range: with b left out, a is the lowest and highest at once
    std::vector<games::position> just_a = {a};
    REQUIRE(games::heat<game_type>(&memory, just_a).of(a, games::colours::floor) == games::colours::hot);
}

TEST_CASE("has_view: the games that can be watched", "[watch][view]")
{
    REQUIRE(games::has_view<games::cliff_walking>::value);
    REQUIRE(games::has_view<games::windy_gridworld>::value);
    REQUIRE(games::has_view<games::dyna_maze>::value);
    REQUIRE(games::has_view<games::frozen_lake>::value);
    REQUIRE(games::has_view<games::racetrack>::value);
    REQUIRE(games::has_view<games::mountain_car>::value);
    REQUIRE(games::has_view<games::cart_pole>::value);
    REQUIRE(games::has_view<games::discretised<games::mountain_car>>::value);
    REQUIRE(games::has_view<games::discretised<games::cart_pole>>::value);
    REQUIRE(!games::has_view<int>::value);
    REQUIRE(!games::has_view<games::position>::value);
}

TEST_CASE("view of Cliff Walking: the cliff, the start, the goal and the agent", "[watch][view]")
{
    using game_type = games::cliff_walking;
    using viewer    = games::view<game_type>;
    REQUIRE(viewer::width() == 48);
    REQUIRE(viewer::height() == 16);
    games::canvas c(viewer::width(), viewer::height());
    game_type game;
    auto s = game.reset();
    viewer::draw(c, game, s, nullptr);
    // the cliff is the bottom row between start and goal: cell (1, 3) is the pixels 4..7 x 12..15
    REQUIRE(c.at(4, 12) == games::colours::danger);
    REQUIRE(c.at(7, 15) == games::colours::danger);
    REQUIRE(c.at(43, 15) == games::colours::danger);
    // the goal, cell (11, 3)
    REQUIRE(c.at(44, 12) == games::colours::goal);
    // the start, cell (0, 3): its corner is the start colour, and the agent sits inside it
    REQUIRE(c.at(0, 12) == games::colours::start);
    REQUIRE(c.at(1, 13) == games::colours::agent);
    REQUIRE(c.at(2, 14) == games::colours::agent);
    REQUIRE(c.at(3, 15) == games::colours::start);
    // the floor
    REQUIRE(c.at(20, 0) == games::colours::floor);
    // the agent moves: up to cell (0, 2)
    game.step(games::up);
    games::canvas d(viewer::width(), viewer::height());
    viewer::draw(d, game, {0, 2}, nullptr);
    REQUIRE(d.at(1, 9) == games::colours::agent);
    REQUIRE(d.at(1, 13) == games::colours::start);
    // with values: the start is coloured by its value, the cliff and the goal stay as they are
    using types = games::types<game_type>;
    types::policy memory;
    memory.update(types::state(games::position{0, 3}), types::action(games::up), -50);
    memory.update(types::state(games::position{0, 2}), types::action(games::up), -10);
    games::canvas e(viewer::width(), viewer::height());
    viewer::draw(e, game, {5, 0}, &memory);
    REQUIRE(e.at(0, 12) == games::colours::cold);
    REQUIRE(e.at(0, 8) == games::colours::hot);
    REQUIRE(e.at(4, 12) == games::colours::danger);
    REQUIRE(e.at(44, 12) == games::colours::goal);
    REQUIRE(e.at(40, 0) == games::colours::floor);
}

TEST_CASE("view of Windy Gridworld: the wind shades the columns", "[watch][view]")
{
    using game_type = games::windy_gridworld;
    using viewer    = games::view<game_type>;
    REQUIRE(viewer::width() == 40);
    REQUIRE(viewer::height() == 28);
    games::canvas c(viewer::width(), viewer::height());
    game_type game;
    viewer::draw(c, game, game.reset(), nullptr);
    // no wind in column 0, the strongest (2) in column 6, half of it in column 3
    REQUIRE(c.at(0, 0) == games::colours::floor);
    REQUIRE(c.at(24, 0) == games::colours::wall);
    REQUIRE(c.at(12, 0) == games::gradient(0.5, games::colours::floor, games::colours::wall));
    // the goal (7, 3) and the start (0, 3), with the agent on it
    REQUIRE(c.at(28, 12) == games::colours::goal);
    REQUIRE(c.at(0, 12) == games::colours::start);
    REQUIRE(c.at(1, 13) == games::colours::agent);
    // with values: the cell (2, 0) is the coldest and the start the hottest; the wind's shade gives way to the value
    using types = games::types<game_type>;
    types::policy memory;
    memory.update(types::state(games::position{2, 0}), types::action(games::up), -30);
    memory.update(types::state(games::position{0, 3}), types::action(games::up), -3);
    games::canvas d(viewer::width(), viewer::height());
    viewer::draw(d, game, {5, 5}, &memory);
    REQUIRE(d.at(8, 0) == games::colours::cold);
    REQUIRE(d.at(0, 12) == games::colours::hot);
    REQUIRE(d.at(24, 0) == games::colours::wall);
    REQUIRE(d.at(28, 12) == games::colours::goal);
}

TEST_CASE("view of the Dyna maze: walls, and a scaled maze drawn at the same size", "[watch][view]")
{
    using game_type = games::dyna_maze;
    using viewer    = games::view<game_type>;
    REQUIRE(viewer::width() == 36);
    REQUIRE(viewer::height() == 24);
    games::canvas c(viewer::width(), viewer::height());
    game_type maze;
    viewer::draw(c, maze, maze.reset(), nullptr);
    // the wall at (2, 1), the start (0, 2) with the agent, the goal (8, 0)
    REQUIRE(c.at(8, 4) == games::colours::wall);
    REQUIRE(c.at(0, 8) == games::colours::start);
    REQUIRE(c.at(1, 9) == games::colours::agent);
    REQUIRE(c.at(32, 0) == games::colours::goal);
    REQUIRE(c.at(0, 0) == games::colours::floor);
    // at scale 2 the maze is 18 x 12, drawn 2 pixels to a cell: the wall is at cells (4..5, 2..5), the start (0, 4)
    game_type big(game_type::maze, 0, 2);
    games::canvas d(viewer::width(), viewer::height());
    viewer::draw(d, big, big.reset(), nullptr);
    REQUIRE(d.at(8, 4) == games::colours::wall);
    REQUIRE(d.at(11, 11) == games::colours::wall);
    REQUIRE(d.at(0, 8) == games::colours::agent);
    REQUIRE(d.at(1, 9) == games::colours::agent);
    REQUIRE(d.at(32, 0) == games::colours::goal);
    // with values, at scale 1: the cell (0, 0) the coldest, the start the hottest, the wall and the goal unchanged
    using types = games::types<game_type>;
    types::policy memory;
    memory.update(types::state(games::position{0, 0}), types::action(games::up), 0.1);
    memory.update(types::state(maze.start()), types::action(games::up), 0.9);
    games::canvas e(viewer::width(), viewer::height());
    viewer::draw(e, maze, {5, 5}, &memory);
    REQUIRE(e.at(0, 0) == games::colours::cold);
    REQUIRE(e.at(0, 8) == games::colours::hot);
    REQUIRE(e.at(8, 4) == games::colours::wall);
    REQUIRE(e.at(32, 0) == games::colours::goal);
    REQUIRE(e.at(4, 0) == games::colours::floor);
}

TEST_CASE("view of FrozenLake: holes, the goal, and the 4 x 4 lake drawn with big cells", "[watch][view]")
{
    using game_type = games::frozen_lake;
    using viewer    = games::view<game_type>;
    REQUIRE(viewer::width() == 32);
    REQUIRE(viewer::height() == 32);
    games::canvas c(viewer::width(), viewer::height());
    game_type lake;
    viewer::draw(c, lake, lake.reset(), nullptr);
    // 8 pixels to a cell: the hole at (1, 1), the goal at (3, 3), the start at (0, 0) with the agent inside
    REQUIRE(c.at(8, 8) == games::colours::danger);
    REQUIRE(c.at(15, 15) == games::colours::danger);
    REQUIRE(c.at(24, 24) == games::colours::goal);
    REQUIRE(c.at(0, 0) == games::colours::start);
    REQUIRE(c.at(1, 1) == games::colours::agent);
    REQUIRE(c.at(6, 6) == games::colours::agent);
    REQUIRE(c.at(7, 7) == games::colours::start);
    REQUIRE(c.at(8, 0) == games::colours::ice);
    // the 8 x 8 lake, 4 pixels to a cell: its first hole is at (3, 2)
    game_type big(game_type::map_8x8());
    games::canvas d(viewer::width(), viewer::height());
    viewer::draw(d, big, 5, nullptr);
    REQUIRE(d.at(12, 8) == games::colours::danger);
    REQUIRE(d.at(28, 28) == games::colours::goal);
    REQUIRE(d.at(21, 1) == games::colours::agent);
    // with values, on the 4 x 4 lake: cell 1 the coldest, cell 14 the hottest, the start, a hole and the goal as they
    // are, and cell 2, without a value, ice
    using types = games::types<game_type>;
    types::policy memory;
    memory.update(types::state(1u), types::action(0u), 0.1);
    memory.update(types::state(14u), types::action(0u), 0.8);
    games::canvas e(viewer::width(), viewer::height());
    viewer::draw(e, lake, 15, &memory);
    REQUIRE(e.at(8, 0) == games::colours::cold);
    REQUIRE(e.at(16, 24) == games::colours::hot);
    REQUIRE(e.at(0, 0) == games::colours::start);
    REQUIRE(e.at(8, 8) == games::colours::danger);
    REQUIRE(e.at(16, 0) == games::colours::ice);
}

TEST_CASE("view of the racetrack: the track, the lines and the car with its velocity", "[watch][view]")
{
    using game_type = games::racetrack;
    using viewer    = games::view<game_type>;
    REQUIRE(viewer::width() == 34);
    REQUIRE(viewer::height() == 64);
    game_type track;
    games::canvas c(viewer::width(), viewer::height());
    // the car at rest on the first start cell, row 31, column 3
    viewer::draw(c, track, track.state_of(31, 3, 0, 0), nullptr);
    REQUIRE(c.at(0, 0) == games::colours::wall);
    REQUIRE(c.at(32, 0) == games::colours::goal);
    REQUIRE(c.at(6, 62) == games::colours::agent);
    REQUIRE(c.at(7, 63) == games::colours::agent);
    REQUIRE(c.at(8, 62) == games::colours::start);
    REQUIRE(c.at(6, 60) == games::colours::floor);
    // moving up by 2 at row 20, column 4: the tail trails down behind the car
    games::canvas d(viewer::width(), viewer::height());
    viewer::draw(d, track, track.state_of(20, 4, 0, 2), nullptr);
    REQUIRE(d.at(8, 40) == games::colours::agent);
    REQUIRE(d.at(8, 43) == games::colours::danger);
    REQUIRE(d.at(8, 45) == games::colours::danger);
    REQUIRE(d.at(8, 46) == games::colours::floor);
    // the finished state draws no car, and lights the finish line up instead: the 'F' cells of rows 0 to 5 in the
    // last column, and nothing else, are the agent's colour
    games::canvas e(viewer::width(), viewer::height());
    viewer::draw(e, track, track.finished(), nullptr);
    for (int x = 0; x < 34; x++) {
        for (int y = 0; y < 64; y++) {
            const bool finish = x >= 32 && y < 12;
            REQUIRE((e.at(x, y) == games::colours::agent) == finish);
        }
    }
    REQUIRE(e.at(32, 0) == games::colours::agent);
    REQUIRE(e.at(33, 11) == games::colours::agent);
    REQUIRE(e.at(32, 12) == games::colours::wall);
    // with values: the cells at rest are coloured, the start cell (31, 3) the coldest and the cell (20, 4) the
    // hottest; a value of a moving car, at (20, 5) with vx = 1, colours nothing
    using types = games::types<game_type>;
    types::policy memory;
    memory.update(types::state(track.state_of(31, 3, 0, 0)), types::action(4u), -20);
    memory.update(types::state(track.state_of(20, 4, 0, 0)), types::action(4u), -5);
    memory.update(types::state(track.state_of(20, 5, 1, 0)), types::action(4u), -1);
    games::canvas f(viewer::width(), viewer::height());
    viewer::draw(f, track, track.state_of(10, 5, 0, 0), &memory);
    REQUIRE(f.at(6, 62) == games::colours::cold);
    REQUIRE(f.at(8, 40) == games::colours::hot);
    REQUIRE(f.at(10, 40) == games::colours::floor);
    REQUIRE(f.at(8, 62) == games::colours::start);
}

TEST_CASE("view of Mountain Car: the hill, the flag and the car", "[watch][view]")
{
    using game_type = games::mountain_car;
    using viewer    = games::view<game_type>;
    REQUIRE(viewer::width() == 64);
    REQUIRE(viewer::height() == 32);
    // the leftmost column is position -1.2, the rightmost 0.6; the goal, 0.5, is near the right
    REQUIRE(viewer::column(-1.2) == 0);
    REQUIRE(viewer::column(0.6) == 63);
    REQUIRE(viewer::column(-0.5) == 25);
    REQUIRE(viewer::column(0.5) == 60);
    // the bottom of the valley is at -pi/6, row 30, the top of the right hill at pi/6, row 10
    REQUIRE(viewer::row(-3.14159265 / 6) == 30);
    REQUIRE(viewer::row(3.14159265 / 6) == 10);
    REQUIRE(viewer::row(-0.5) == 30);
    REQUIRE(viewer::row(-1.2) == 16);
    REQUIRE(viewer::row(0.5) == 10);
    games::canvas c(viewer::width(), viewer::height());
    game_type game;
    viewer::draw(c, game, {-0.5, 0.0}, nullptr);
    // the car sits on the hill at its position, 3 x 2 pixels above the ground
    REQUIRE(c.at(25, 29) == games::colours::agent);
    REQUIRE(c.at(24, 28) == games::colours::agent);
    REQUIRE(c.at(26, 29) == games::colours::agent);
    REQUIRE(c.at(25, 30) == games::colours::ground);
    REQUIRE(c.at(25, 31) == games::colours::ground);
    REQUIRE(c.at(25, 20) == games::colours::sky);
    // the ground follows the hill
    REQUIRE(c.at(0, 16) == games::colours::ground);
    REQUIRE(c.at(0, 15) == games::colours::sky);
    // the flag at the goal: a pole from row 4 to row 9, and a flag of 3 x 2 at its top
    REQUIRE(c.at(60, 7) == games::colours::white);
    REQUIRE(c.at(60, 9) == games::colours::white);
    REQUIRE(c.at(60, 3) == games::colours::sky);
    REQUIRE(c.at(61, 4) == games::colours::goal);
    REQUIRE(c.at(63, 5) == games::colours::goal);
    REQUIRE(c.at(61, 6) == games::colours::sky);
}

TEST_CASE("view of Mountain Car cut into a grid: the hill, and the plane of values", "[watch][view]")
{
    using game_type = games::discretised<games::mountain_car>;
    using viewer    = games::view<game_type>;
    using types     = games::types<game_type>;
    REQUIRE(viewer::width() == 129);
    REQUIRE(viewer::height() == 32);
    game_type game(games::mountain_car(1), {{-1.2, 0.6, 20}, {-0.07, 0.07, 20}});
    game.place({-0.5, 0.0});
    const unsigned int cell = game.cell({-0.5, 0.0});
    // the hill on the left, as before
    games::canvas c(viewer::width(), viewer::height());
    viewer::draw(c, game, cell, nullptr);
    REQUIRE(c.at(25, 29) == games::colours::agent);
    // without values, the plane is floor, with the car's cell on it: position -0.5 is 0.39 of the way, velocity
    // 0 is half way down
    REQUIRE(c.at(65, 0) == games::colours::floor);
    REQUIRE(c.at(128, 31) == games::colours::floor);
    REQUIRE(c.at(64, 0) == games::colours::black);
    // the car's pixel: position -0.5 is 0.389 of the way, pixel 24 of 64, and velocity 0 half way down, pixel 16;
    // the marker is the 3 x 3 square round it
    REQUIRE(viewer::plane_column(game, -0.5) == 24);
    REQUIRE(viewer::plane_row(game, 0.0) == 16);
    REQUIRE(viewer::plane_column(game, -1.2) == 0);
    REQUIRE(viewer::plane_column(game, 0.6) == 63);
    REQUIRE(viewer::plane_row(game, 0.07) == 0);
    REQUIRE(viewer::plane_row(game, -0.07) == 31);
    REQUIRE(c.at(65 + 24, 16) == games::colours::agent);
    REQUIRE(c.at(65 + 23, 15) == games::colours::agent);
    REQUIRE(c.at(65 + 25, 17) == games::colours::agent);
    REQUIRE(c.at(65 + 22, 15) == games::colours::floor);
    REQUIRE(c.at(65 + 26, 16) == games::colours::floor);
    // with values: the top-left pixel of the plane is the cell of the lowest position and the highest velocity, 19
    types::policy memory;
    memory.update(types::state(19u), types::action(0u), -5);
    memory.update(types::state(cell), types::action(0u), -50);
    games::canvas d(viewer::width(), viewer::height());
    viewer::draw(d, game, cell, &memory);
    REQUIRE(d.at(65, 0) == games::colours::hot);
    // the bottom-right pixel is cell 380 (position bin 19, velocity bin 0), which has no value
    REQUIRE(d.at(128, 31) == games::colours::floor);
    // the car's cell (position bin 7, velocity bin 10), where it is not covered by the car, is the coldest: the
    // pixel (22, 15) of the plane, just left of the marker, is still in that cell
    REQUIRE(game.cell({-0.5, 0.0}) == 7 * 20 + 10);
    REQUIRE(game.cell({-1.2 + 22.5 / 64 * 1.8, 0.07 - 15.5 / 32 * 0.14}) == 7 * 20 + 10);
    REQUIRE(d.at(65 + 22, 15) == games::colours::cold);
}

TEST_CASE("view of cart-pole: the track, the cart and the pole", "[watch][view]")
{
    using game_type = games::cart_pole;
    using viewer    = games::view<game_type>;
    REQUIRE(viewer::width() == 64);
    REQUIRE(viewer::height() == 24);
    games::canvas c(viewer::width(), viewer::height());
    game_type game;
    // upright in the middle: the cart at column 32, the pole straight up from it
    viewer::draw(c, game, {0.0, 0.0, 0.0, 0.0}, nullptr);
    REQUIRE(c.at(0, 0) == games::colours::sky);
    REQUIRE(c.at(0, 22) == games::colours::ground);
    REQUIRE(c.at(32, 20) == games::colours::wall);
    REQUIRE(c.at(29, 19) == games::colours::wall);
    REQUIRE(c.at(35, 21) == games::colours::wall);
    REQUIRE(c.at(28, 20) == games::colours::sky);
    REQUIRE(c.at(32, 10) == games::colours::agent);
    REQUIRE(c.at(32, 5) == games::colours::agent);
    REQUIRE(c.at(32, 4) == games::colours::sky);
    REQUIRE(c.at(31, 10) == games::colours::sky);
    // leaning right by 0.2 rad: the tip is 3 pixels to the right (14 sin 0.2 = 2.8)
    games::canvas d(viewer::width(), viewer::height());
    viewer::draw(d, game, {0.0, 0.0, 0.2, 0.0}, nullptr);
    REQUIRE(d.at(35, 19 - 14) == games::colours::agent);
    REQUIRE(d.at(32, 5) == games::colours::sky);
    // the cart at the left end of the track
    games::canvas e(viewer::width(), viewer::height());
    viewer::draw(e, game, {-2.4, 0.0, 0.0, 0.0}, nullptr);
    REQUIRE(e.at(0, 20) == games::colours::wall);
    REQUIRE(e.at(3, 20) == games::colours::wall);
    REQUIRE(e.at(4, 20) == games::colours::sky);
}

TEST_CASE("view of any other game cut into a grid: that of the game inside", "[watch][view]")
{
    using game_type = games::discretised<games::cart_pole>;
    using viewer    = games::view<game_type>;
    REQUIRE(viewer::width() == games::view<games::cart_pole>::width());
    REQUIRE(viewer::height() == games::view<games::cart_pole>::height());
    game_type game(games::cart_pole(1), {{-0.8, 0.8, 3}, {-0.5, 0.5, 3}, {-0.21, 0.21, 6}, {-0.87, 0.87, 3}});
    game.place({0.0, 0.0, 0.0, 0.0});
    games::canvas c(viewer::width(), viewer::height());
    viewer::draw(c, game, 0, nullptr);
    games::canvas d(viewer::width(), viewer::height());
    games::view<games::cart_pole>::draw(d, game.inner(), game.raw(), nullptr);
    for (int x = 0; x < static_cast<int>(c.width()); x++) {
        for (int y = 0; y < static_cast<int>(c.height()); y++) {
            REQUIRE(c.at(x, y) == d.at(x, y));
        }
    }
}

TEST_CASE("watched: plays exactly as the game it wraps", "[watch][watched]")
{
    using game_type = games::cliff_walking;
    game_type plain;
    std::ostringstream out;
    games::watched<game_type> shown(game_type(), out);
    shown.delay = 0;
    for (bool on : {false, true}) {
        shown.on = on;
        random_agent<game_type> first{std::mt19937(7)};
        random_agent<game_type> second{std::mt19937(7)};
        auto a = games::play(plain, first, 60);
        auto b = games::play(shown, second, 60);
        REQUIRE(a.steps == b.steps);
        REQUIRE(a.total_reward == b.total_reward);
        REQUIRE(a.terminated == b.terminated);
        REQUIRE(a.episode.size() == b.episode.size());
        for (std::size_t i = 0; i < a.episode.size(); i++) {
            REQUIRE(a.episode[i].state == b.episode[i].state);
            REQUIRE(a.episode[i].state.reward() == b.episode[i].state.reward());
            REQUIRE(a.episode[i].action == b.episode[i].action);
        }
        REQUIRE(shown.actions({3, 3}) == plain.actions({3, 3}));
    }
    // and learning through it gives the same values
    using types = games::types<game_type>;
    types::policy from_plain;
    types::policy from_shown;
    relearn::sarsa<types::state, types::action> learner;
    learner.alpha = 0.5;
    game_type again;
    games::watched<game_type> again_shown(game_type(), out);
    again_shown.delay = 0;
    std::mt19937 p(3);
    std::mt19937 q(3);
    auto agent_plain = [&](const games::position & s, const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(from_plain, s, legal, 0.1, p);
    };
    auto agent_shown = [&](const games::position & s, const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<games::watched<game_type>>(from_shown, s, legal, 0.1, q);
    };
    for (int i = 0; i < 20; i++) {
        games::play_online(again, agent_plain, learner, from_plain, 200);
        games::play_online(again_shown, agent_shown, learner, from_shown, 200);
    }
    for (int x = 0; x < game_type::width; x++) {
        for (int y = 0; y < game_type::height; y++) {
            for (unsigned int a : {games::up, games::right, games::down, games::left}) {
                REQUIRE(from_plain.value(types::state(games::position{x, y}), types::action(a)) ==
                        from_shown.value(types::state(games::position{x, y}), types::action(a)));
            }
        }
    }
}

TEST_CASE("watched: counts the episodes, their steps and their returns", "[watch][watched]")
{
    using game_type = games::cliff_walking;
    std::ostringstream out;
    games::watched<game_type> game(game_type(), out);
    game.on = false;
    REQUIRE(game.episode() == 0);
    REQUIRE(game.returns().empty());
    game.reset();
    REQUIRE(game.episode() == 1);
    REQUIRE(game.steps() == 0);
    REQUIRE(game.total() == 0);
    // up, then right: -1 each
    game.step(games::up);
    game.step(games::right);
    REQUIRE(game.steps() == 2);
    REQUIRE(game.total() == -2);
    // down, into the cliff: -100, back to the start
    auto out_step = game.step(games::down);
    REQUIRE(out_step.reward == -100);
    REQUIRE(out_step.next == game_type::start());
    REQUIRE(game.total() == -102);
    // a new episode closes this one
    game.reset();
    REQUIRE(game.episode() == 2);
    REQUIRE(game.steps() == 0);
    REQUIRE(game.total() == 0);
    REQUIRE(game.returns() == std::vector<double>{-102});
    game.step(games::up);
    game.reset();
    REQUIRE(game.returns() == std::vector<double>{-102, -1});
    // an episode that ends is recorded as it ends, so that its last frame shows it, and not again at the next reset
    for (int i = 0; i < 3; i++) {
        game.step(games::up);
    }
    for (int i = 0; i < 11; i++) {
        game.step(games::right);
    }
    game.step(games::down);
    game.step(games::down);
    REQUIRE(game.returns().size() == 2);
    auto last = game.step(games::down);
    REQUIRE(last.terminated);
    REQUIRE(game.returns() == std::vector<double>{-102, -1, -17});
    game.reset();
    REQUIRE(game.returns() == std::vector<double>{-102, -1, -17});
    REQUIRE(game.episode() == 4);
    // nothing was written: it was off
    REQUIRE(out.str().empty());
}

TEST_CASE("watched: draws a frame at the start and after every step, and only when on", "[watch][watched]")
{
    using game_type = games::cliff_walking;
    std::string written;
    {
        std::ostringstream out;
        games::watched<game_type> game(game_type(), out);
        game.delay = 0;
        game.title = "cliff";
        game.reset();
        game.step(games::up);
        game.step(games::right);
        const std::string text = out.str();
        // the cursor is hidden and the screen cleared once, then one frame per reset and step, each from the
        // top-left corner, with the status line and the rest of the screen cleared after it
        REQUIRE(text.find(games::terminal::hide_cursor()) == 0);
        REQUIRE(count(text, games::terminal::hide_cursor()) == 1);
        REQUIRE(count(text, "\x1b[2J") == 1);
        REQUIRE(count(text, games::terminal::clear_below()) == 3);
        // the frame: 16 rows of the game, a blank row and 16 of the curve are 33 pixel rows, 17 lines, and the
        // status line, which ends with a newline so that whatever is printed next goes below it
        REQUIRE(game.lines() == 18);
        REQUIRE(count(text, "\n") == 3 * 18);
        REQUIRE(text.find("cliff  episode 1  step 0  return 0\n") != std::string::npos);
        REQUIRE(text.find("cliff  episode 1  step 2  return -2\n" + std::string(games::terminal::clear_below())) !=
                std::string::npos);
        REQUIRE(text.find(games::terminal::show_cursor()) == std::string::npos);
        // switched off, nothing more is written
        game.on = false;
        game.step(games::right);
        REQUIRE(out.str() == text);
        game.on = true;
        game.step(games::right);
        REQUIRE(count(out.str(), games::terminal::clear_below()) == 4);
        written = out.str();
        // the frame is drawn with the terminal's colours
        if (games::truecolour::available()) {
            REQUIRE(written.find("\x1b[38;2;") != std::string::npos);
        }
        else {
            REQUIRE(written.find("\x1b[38;5;") != std::string::npos);
        }
        // the status line is also given on its own
        REQUIRE(game.status() == "cliff  episode 1  step 4  return -4");
    }
    // the cursor is shown again when the watched game goes
    (void) written;
    {
        std::ostringstream out;
        {
            games::watched<game_type> game(game_type(), out);
            game.delay = 0;
            game.reset();
        }
        const std::string text = out.str();
        REQUIRE(text.size() > std::strlen(games::terminal::show_cursor()) + 1);
        REQUIRE(text.compare(text.size() - std::strlen(games::terminal::show_cursor()) - 1,
                             std::strlen(games::terminal::show_cursor()), games::terminal::show_cursor()) == 0);
        REQUIRE(text.back() == '\n');
    }
    // but not when nothing was ever drawn
    {
        std::ostringstream out;
        {
            games::watched<game_type> game(game_type(), out);
            game.on = false;
            game.reset();
            game.step(games::up);
        }
        REQUIRE(out.str().empty());
    }
}

TEST_CASE("watched: every n-th step, the end of the episode always, and no curve", "[watch][watched]")
{
    using game_type = games::cliff_walking;
    std::ostringstream out;
    games::watched<game_type> game(game_type(), out);
    game.delay = 0;
    game.every = 3;
    game.curve_height = 0;
    game.reset();
    // the start is drawn; then steps 1 and 2 are not, step 3 is
    REQUIRE(count(out.str(), games::terminal::home()) == 1);
    game.step(games::up);
    game.step(games::up);
    REQUIRE(count(out.str(), games::terminal::home()) == 1);
    game.step(games::up);
    REQUIRE(count(out.str(), games::terminal::home()) == 2);
    // without the curve a frame is the game's 16 pixel rows alone, 8 lines, and the status line
    REQUIRE(game.lines() == 9);
    REQUIRE(count(out.str(), "\n") == 2 * 9);
    // walk to the goal along the top: 11 moves right, then 3 down; the last step ends the episode and is drawn
    // whatever `every` says
    for (int i = 0; i < 11; i++) {
        game.step(games::right);
    }
    game.step(games::down);
    game.step(games::down);
    const std::size_t before = count(out.str(), games::terminal::home());
    auto last = game.step(games::down);
    REQUIRE(last.terminated);
    REQUIRE(game.steps() == 17);
    REQUIRE(count(out.str(), games::terminal::home()) == before + 1);
    // `every` set to 0 by hand counts as 1
    game.every = 0;
    game.reset();
    const std::size_t start = count(out.str(), games::terminal::home());
    game.step(games::up);
    REQUIRE(count(out.str(), games::terminal::home()) == start + 1);
}

TEST_CASE("watched: the frame is the view, the curve under it and the status, in the terminal's colours", "[watch][watched]")
{
    using game_type = games::cliff_walking;
    using types     = games::types<game_type>;
    using viewer    = games::view<game_type>;
    types::policy memory;
    memory.update(types::state(games::position{0, 3}), types::action(games::up), -50);
    memory.update(types::state(games::position{0, 2}), types::action(games::up), -10);
    std::ostringstream out;
    games::watched<game_type> game(game_type(), out);
    game.delay  = 0;
    game.memory = &memory;
    game.on     = false;
    // an episode of 1 step, cut short, then a new one that has moved up: the state is (0, 2) and the curve has -1
    game.reset();
    game.step(games::up);
    game.reset();
    game.step(games::up);
    for (const char * colorterm : {"truecolor", "xterm"}) {
        setenv("COLORTERM", colorterm, 1);
        // the same, drawn by hand: the view with the values, a blank row, and the curve of the returns
        games::canvas c(viewer::width(), viewer::height() + 1 + 16);
        viewer::draw(c, game.inner(), {0, 2}, &memory);
        games::plot_curve(c, 0, 17, 48, 16, {-1.0});
        const std::string expected = games::truecolour::available(colorterm) ? c.render<games::truecolour>()
                                                                             : c.render<games::colour_256>();
        REQUIRE(game.picture() == expected);
        // the frame is the picture, then the status line
        game.on = true;
        game.frame();
        const std::string text = out.str();
        REQUIRE(text.compare(text.size() - expected.size() - std::strlen("episode 2  step 1  return -1\n\x1b[J"),
                             expected.size(), expected) == 0);
        REQUIRE(text.find("episode 2  step 1  return -1\n") != std::string::npos);
        game.on = false;
    }
    unsetenv("COLORTERM");
    // the two palettes give different text
    setenv("COLORTERM", "truecolor", 1);
    const std::string full = game.picture();
    unsetenv("COLORTERM");
    REQUIRE(game.picture() != full);
    REQUIRE(full.find("\x1b[38;2;") != std::string::npos);
    REQUIRE(game.picture().find("\x1b[38;5;") != std::string::npos);
    REQUIRE(game.picture().find("\x1b[38;2;") == std::string::npos);
}

TEST_CASE("watched: fits the terminal, dropping the curve, or says so", "[watch][watched]")
{
    using game_type = games::cliff_walking;
    // 18 lines with the curve fit a terminal of 20 rows
    {
        std::ostringstream out;
        games::watched<game_type> game(game_type(), out);
        game.delay = 0;
        game.terminal_rows = 20;
        game.reset();
        REQUIRE(game.curve_height == 16);
        REQUIRE(game.lines() == 18);
        REQUIRE(out.str().find("the picture needs") == std::string::npos);
    }
    // not one of 12: the curve goes, and the 9 lines left fit
    {
        std::ostringstream out;
        games::watched<game_type> game(game_type(), out);
        game.delay = 0;
        game.terminal_rows = 12;
        game.reset();
        REQUIRE(game.curve_height == 0);
        REQUIRE(game.lines() == 9);
        REQUIRE(out.str().find("the picture needs") == std::string::npos);
    }
    // nor one of 5: the curve goes, and a note is written once, before the frames
    {
        std::ostringstream out;
        games::watched<game_type> game(game_type(), out);
        game.delay = 0;
        game.terminal_rows = 5;
        game.reset();
        game.step(games::up);
        REQUIRE(game.curve_height == 0);
        REQUIRE(count(out.str(), "the picture needs 9 rows, and the terminal has 5") == 1);
        REQUIRE(out.str().find("the picture needs") < out.str().find(games::terminal::hide_cursor()));
    }
    // 0 is not checked
    {
        std::ostringstream out;
        games::watched<game_type> game(game_type(), out);
        game.delay = 0;
        game.reset();
        REQUIRE(game.curve_height == 16);
    }
}

TEST_CASE("watched: the picture holds the game and the curve of the returns", "[watch][watched]")
{
    using game_type = games::cliff_walking;
    std::ostringstream out;
    games::watched<game_type> game(game_type(), out);
    game.on = false;
    game.reset();
    game.step(games::up);
    game.reset();
    game.step(games::up);
    game.step(games::up);
    game.reset();
    REQUIRE(game.returns() == std::vector<double>{-1, -2});
    // 16 rows of the game, 1 blank, 16 of the curve: 33 pixel rows, so 17 lines
    const std::string picture = game.picture();
    REQUIRE(count(picture, "\n") == 17);
    REQUIRE(count(picture, "\xE2\x96\x80") == 17 * 48);
    // the curve is there: its axis and its line are drawn, in the terminal's palette
    const bool truecolour = games::truecolour::available();
    const std::string axis  = truecolour ? games::truecolour::foreground(games::colours::axis)
                                         : games::colour_256::foreground(games::colours::axis);
    const std::string curve = truecolour ? games::truecolour::foreground(games::colours::curve)
                                         : games::colour_256::foreground(games::colours::curve);
    const std::string floor = truecolour ? games::truecolour::foreground(games::colours::floor)
                                         : games::colour_256::foreground(games::colours::floor);
    REQUIRE(picture.find(axis) != std::string::npos);
    REQUIRE(picture.find(curve) != std::string::npos);
    REQUIRE(picture.find(floor) != std::string::npos);
    // without it, the game alone
    game.curve_height = 0;
    const std::string alone = game.picture();
    REQUIRE(count(alone, "\n") == 8);
    REQUIRE(alone.find(axis) == std::string::npos);
    REQUIRE(alone.find(curve) == std::string::npos);
    REQUIRE(alone.find(floor) != std::string::npos);
    // the inner game is reachable
    REQUIRE(game.inner().actions({0, 0}).size() == 4);
    const games::watched<game_type> & fixed = game;
    REQUIRE(fixed.inner().actions({0, 0}).size() == 4);
}

TEST_CASE("watch_options: the command line", "[watch][options]")
{
    const char * none[] = {"prog"};
    auto defaults = games::watch_options::parse(1, const_cast<char **>(none));
    REQUIRE(!defaults.on);
    REQUIRE(defaults.delay == 30);
    REQUIRE(defaults.every == 1);
    const char * all[] = {"prog", "--watch", "--delay", "5", "--every", "3"};
    auto given = games::watch_options::parse(6, const_cast<char **>(all));
    REQUIRE(given.on);
    REQUIRE(given.delay == 5);
    REQUIRE(given.every == 3);
    // `--every 0` is 1, a missing value is ignored, and unknown words are too
    const char * odd[] = {"prog", "--every", "0", "--what", "--delay"};
    auto parsed = games::watch_options::parse(5, const_cast<char **>(odd));
    REQUIRE(!parsed.on);
    REQUIRE(parsed.every == 1);
    REQUIRE(parsed.delay == 30);
    // a negative delay is none
    const char * negative[] = {"prog", "--delay", "-5"};
    REQUIRE(games::watch_options::parse(3, const_cast<char **>(negative)).delay == 0);
    // applied to a watched game, which is also told the terminal's height
    std::ostringstream out;
    games::watched<games::cliff_walking> game(games::cliff_walking(), out);
    given.apply(game);
    REQUIRE(game.on);
    REQUIRE(game.delay == 5);
    REQUIRE(game.every == 3);
    REQUIRE(game.terminal_rows == games::terminal::rows());
    defaults.apply(game);
    REQUIRE(!game.on);
}
