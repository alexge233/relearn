/**
 * Tests of the FrozenLake lesson's layer: the lake as a grid, what is said about the ice, and a session that plays the
 * lake's own rules, tells a hole from the treasure, and learns what the lake's model says is best.
 */
#include "../examples/window/frozen.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

namespace lessons = games::lessons;
using lessons::frozen_grid;
using games::position;
using q_session = lessons::frozen_session<lessons::q_learning_rule<frozen_grid>>;

namespace
{
const std::vector<std::string> names = {"up", "right", "down", "left"};

/// @brief the formula on the screen must hold at every step, against the value in the policy
template <class rule>
void learns_the_formula()
{
    lessons::frozen_session<rule> s(frozen_grid(3), 0.2, 0.9, 0.2, 3, 100);
    for (int i = 0; i < 30000; i++) {
        s.step();
        const auto & n = s.latest();
        REQUIRE(n.valid);
        REQUIRE(n.learnt == Approx(n.believed + 0.2 * (n.reward + (n.ended ? 0.0 : 0.9 * n.next) - n.believed)).margin(1e-12));
        REQUIRE(s.q(s.from(), s.last_action()) == n.learnt);
    }
}

/// @return the share of @param games won by following the best known moves of @param s
template <class session_type>
double greedy_wins(const session_type & s,
                   int games_played)
{
    frozen_grid lake(99);
    int wins = 0;
    for (int g = 0; g < games_played; g++) {
        position at = lake.reset();
        for (int i = 0; i < 100; i++) {
            const auto out = lake.step(s.best_action(at));
            at = out.next;
            if (out.terminated) {
                wins += out.reward > 0;
                break;
            }
        }
    }
    return static_cast<double>(wins) / games_played;
}
} // namespace

TEST_CASE("the lake as a grid: cells and tiles", "[frozen]")
{
    frozen_grid lake;
    REQUIRE(lake.rows() == 4);
    REQUIRE(lake.columns() == 4);
    REQUIRE(lake.reset() == position{0, 0});
    REQUIRE(lake.tile({0, 0}) == 'S');
    REQUIRE(lake.tile({1, 1}) == 'H');
    REQUIRE(lake.tile({3, 1}) == 'H');
    REQUIRE(lake.tile({3, 2}) == 'H');
    REQUIRE(lake.tile({0, 3}) == 'H');
    REQUIRE(lake.tile({3, 3}) == 'G');
    REQUIRE(lake.tile({2, 1}) == 'F');
    REQUIRE(lake.actions({0, 0}) == std::vector<unsigned int>{0, 1, 2, 3});
}

TEST_CASE("the lake as a grid: each move goes its own way, or to either side, a third of the time each", "[frozen]")
{
    // from the middle of a cell with four free neighbours
    const position from{2, 2};
    const std::vector<std::vector<position>> allowed = {{{2, 1}, {3, 2}, {1, 2}}, {{3, 2}, {2, 1}, {2, 3}}, {{2, 3}, {3, 2}, {1, 2}}, {{1, 2}, {2, 1}, {2, 3}}};
    frozen_grid lake(4);
    for (unsigned int a = 0; a < 4; a++) {
        std::vector<int> count(3, 0);
        const int tries = 30000;
        for (int i = 0; i < tries; i++) {
            lake.place(from);
            const auto out = lake.step(a);
            const auto found = std::find(allowed[a].begin(), allowed[a].end(), out.next);
            REQUIRE(found != allowed[a].end());
            count[static_cast<std::size_t>(found - allowed[a].begin())]++;
        }
        for (int c : count) {
            REQUIRE(static_cast<double>(c) / tries == Approx(1.0 / 3).margin(0.02));
        }
    }
}

TEST_CASE("the lake as a grid: reward and the end of the game", "[frozen]")
{
    frozen_grid lake(5);
    int goals = 0;
    int holes = 0;
    for (int i = 0; i < 300; i++) {
        lake.place({2, 3});
        const auto out = lake.step(1);
        if (out.next == position{3, 3}) {
            REQUIRE(out.reward == 1);
            REQUIRE(out.terminated);
            goals++;
        }
        else {
            REQUIRE(out.reward == 0);
            REQUIRE(!out.terminated);
        }
        lake.place({1, 0});
        const auto down = lake.step(2);
        if (down.next == position{1, 1}) {
            REQUIRE(down.reward == 0);
            REQUIRE(down.terminated);
            holes++;
        }
    }
    REQUIRE(goals > 50);
    REQUIRE(holes > 50);
}

TEST_CASE("the edge keeps a move on the board", "[frozen]")
{
    REQUIRE(lessons::on_board({-1, 2}, 4, 4) == position{0, 2});
    REQUIRE(lessons::on_board({4, 2}, 4, 4) == position{3, 2});
    REQUIRE(lessons::on_board({2, -1}, 4, 4) == position{2, 0});
    REQUIRE(lessons::on_board({2, 4}, 4, 4) == position{2, 3});
    REQUIRE(lessons::on_board({1, 2}, 4, 4) == position{1, 2});
}

TEST_CASE("a move turned to either side", "[frozen]")
{
    REQUIRE(lessons::turned(0, -1) == 3);
    REQUIRE(lessons::turned(0, 0) == 0);
    REQUIRE(lessons::turned(0, 1) == 1);
    REQUIRE(lessons::turned(3, 1) == 0);
    REQUIRE(lessons::turned(2, -1) == 1);
}

TEST_CASE("which way the ice turned a move", "[frozen]")
{
    const position from{1, 1};
    REQUIRE(lessons::turn_of(from, 1, {2, 1}, 4, 4) == 0);
    REQUIRE(lessons::turn_of(from, 1, {1, 0}, 4, 4) == -1);
    REQUIRE(lessons::turn_of(from, 1, {1, 2}, 4, 4) == 1);
    REQUIRE(lessons::turn_of(from, 0, {0, 1}, 4, 4) == -1);
    REQUIRE(lessons::turn_of(from, 0, {2, 1}, 4, 4) == 1);
    // into the edge: choosing up on the top row and staying is as chosen
    REQUIRE(lessons::turn_of({1, 0}, 0, {1, 0}, 4, 4) == 0);
    // in a corner two moves stay put: as chosen comes first
    REQUIRE(lessons::turn_of({0, 0}, 0, {0, 0}, 4, 4) == 0);
    // sliding left into the edge when it chose up
    REQUIRE(lessons::turn_of({0, 1}, 0, {0, 1}, 4, 4) == -1);
}

TEST_CASE("the way a move goes when it goes as chosen: a straight line", "[frozen][slide]")
{
    const position from{1, 1};
    const position to{2, 1};
    const auto middle = lessons::slide_path(from, 1, to, 0, 0.5);
    REQUIRE(middle.x == Approx(1.5));
    REQUIRE(middle.y == Approx(1));
    REQUIRE(lessons::slide_path(from, 1, to, 0, 0).x == Approx(1));
    REQUIRE(lessons::slide_path(from, 1, to, 0, 1).x == Approx(2));
}

TEST_CASE("the way a slip goes: off as chosen, then the ice takes it", "[frozen][slide]")
{
    // it chose right, and the ice sent it down
    const position from{1, 1};
    const position to{1, 2};
    const auto start = lessons::slide_path(from, 1, to, 1, 0);
    REQUIRE(start.x == Approx(1));
    REQUIRE(start.y == Approx(1));
    // on the way out it only moves the way it chose
    const auto out = lessons::slide_path(from, 1, to, 1, 0.2);
    REQUIRE(out.x == Approx(1.15));
    REQUIRE(out.y == Approx(1));
    // the turn: as far as it gets
    const auto pivot = lessons::slide_path(from, 1, to, 1, 0.4);
    REQUIRE(pivot.x == Approx(1.3));
    REQUIRE(pivot.y == Approx(1));
    // no jump as the ice takes over
    const auto after = lessons::slide_path(from, 1, to, 1, 0.4001);
    REQUIRE(after.x == Approx(pivot.x).margin(0.001));
    REQUIRE(after.y == Approx(pivot.y).margin(0.001));
    const auto end = lessons::slide_path(from, 1, to, 1, 1);
    REQUIRE(end.x == Approx(1));
    REQUIRE(end.y == Approx(2));
    // kept in its range
    REQUIRE(lessons::slide_path(from, 1, to, 1, 5).y == Approx(2));
    REQUIRE(lessons::slide_path(from, 1, to, 1, -1).x == Approx(1));
}

TEST_CASE("a slip into the edge goes out and comes back", "[frozen][slide]")
{
    const position from{0, 1};
    const auto pivot = lessons::slide_path(from, 0, from, -1, 0.4);
    REQUIRE(pivot.y == Approx(0.7));
    REQUIRE(pivot.x == Approx(0));
    const auto end = lessons::slide_path(from, 0, from, -1, 1);
    REQUIRE(end.y == Approx(1));
    REQUIRE(end.x == Approx(0));
}

TEST_CASE("the animation of a step, paused or not", "[frozen][slide]")
{
    lessons::controls c;
    c.slow_down_to(2);
    REQUIRE(c.rate() == 2);
    c.steps_for(0.1);
    REQUIRE(c.animation() == Approx(0.1 / lessons::controls::longest_animation));
    // the step that comes due starts it again
    REQUIRE(c.steps_for(0.4) == 1);
    REQUIRE(c.animation() == 0);
    // and it stops at the end, however long the step lasts
    c.paused = true;
    c.steps_for(1);
    REQUIRE(c.animation() == 1);
    // one step asked for by hand has its animation too
    c.single_step();
    REQUIRE(c.steps_for(0.01) == 1);
    REQUIRE(c.animation() == 0);
    c.steps_for(0.225);
    REQUIRE(c.animation() == Approx(0.5));
    // at a speed where a step is shorter, so is the animation
    c.paused = false;
    while (c.rate() < 10) {
        c.faster();
    }
    c.steps_for(0.2);
    REQUIRE(c.animation() == 0);
    c.steps_for(0.05);
    REQUIRE(c.animation() == Approx(0.5).margin(0.11));
}

TEST_CASE("a move that goes as chosen is a little hop, upright", "[frozen][pose]")
{
    REQUIRE(lessons::step_pose(0, 0).lift == Approx(0).margin(1e-9));
    REQUIRE(lessons::step_pose(0, 0.5).lift == Approx(0.09));
    REQUIRE(lessons::step_pose(0, 1).lift == Approx(0).margin(1e-9));
    for (double t : {0.0, 0.3, 0.7, 1.0}) {
        REQUIRE(lessons::step_pose(0, t).angle == 0);
        REQUIRE(lessons::step_pose(0, t).size == 1);
    }
}

TEST_CASE("a slip keels over the way it skids, and comes back", "[frozen][pose]")
{
    // upright whilst it sets off the way it chose
    REQUIRE(lessons::step_pose(1, 0).angle == 0);
    REQUIRE(lessons::step_pose(1, 0.39).angle == 0);
    REQUIRE(lessons::step_pose(1, 0.39).lift == 0);
    // from the start of the skid: no jump
    REQUIRE(lessons::step_pose(1, 0.4).angle == Approx(0).margin(1e-9));
    // furthest over half way through the skid, one way or the other
    REQUIRE(lessons::step_pose(1, 0.7).angle == Approx(55).margin(1e-9));
    REQUIRE(lessons::step_pose(-1, 0.7).angle == Approx(-55).margin(1e-9));
    REQUIRE(lessons::step_pose(1, 0.7).lift == Approx(0.12));
    // the wobble on top of it
    REQUIRE(lessons::step_pose(1, 0.475).angle == Approx(55 * std::sin(std::acos(-1.0) / 8) + 12));
    // and back on its feet
    REQUIRE(lessons::step_pose(1, 1).angle == Approx(0).margin(1e-9));
    REQUIRE(lessons::step_pose(1, 1).lift == Approx(0).margin(1e-9));
    REQUIRE(lessons::step_pose(1, 2).angle == Approx(0).margin(1e-9));
    REQUIRE(lessons::step_pose(1, 0.7).size == 1);
}

TEST_CASE("going into a hole, it spins and shrinks", "[frozen][pose]")
{
    REQUIRE(lessons::sinking_pose(0).angle == 0);
    REQUIRE(lessons::sinking_pose(0).size == 1);
    REQUIRE(lessons::sinking_pose(0.5).angle == Approx(270));
    REQUIRE(lessons::sinking_pose(1).angle == Approx(540));
    REQUIRE(lessons::sinking_pose(1).size == Approx(0.25));
    REQUIRE(lessons::sinking_pose(3).size == Approx(0.25));
    REQUIRE(lessons::sinking_pose(-1).size == 1);
}

TEST_CASE("slowing down to a speed", "[frozen][slide]")
{
    lessons::controls c;
    REQUIRE(c.rate() == 5);
    c.slow_down_to(5);
    REQUIRE(c.rate() == 5);
    c.slow_down_to(100);
    REQUIRE(c.rate() == 5);
    c.slow_down_to(2);
    REQUIRE(c.rate() == 2);
    c.slow_down_to(0.5);
    REQUIRE(c.rate() == 1);
}

TEST_CASE("what is said about the ice", "[frozen][words]")
{
    REQUIRE(lessons::slip_words(names, {1, 1}, 1, {2, 1}, 4, 4).empty());
    REQUIRE(lessons::slip_words(names, {1, 1}, 1, {1, 2}, 4, 4) == "It tried going right, but the ice sent it down.");
    REQUIRE(lessons::slip_words(names, {1, 1}, 0, {0, 1}, 4, 4) == "It tried going up, but the ice sent it left.");
    REQUIRE(lessons::slip_words(names, {0, 1}, 0, {0, 1}, 4, 4) == "It tried going up, but slid into the edge and stayed where it was.");
    REQUIRE(lessons::slip_words(names, {1, 0}, 0, {1, 0}, 4, 4).empty());
}

TEST_CASE("a hole is a fall and the treasure is not, and only they end an episode with a mark", "[frozen]")
{
    q_session s(frozen_grid(2), 0.1, 0.95, 0.5, 2, 100);
    frozen_grid lake;
    int falls = 0;
    int wins = 0;
    for (int i = 0; i < 40000; i++) {
        s.step();
        const char tile = lake.tile(s.at());
        REQUIRE(s.punished() == (tile == 'H'));
        REQUIRE(s.latest().ended == (tile == 'H' || tile == 'G'));
        falls += tile == 'H';
        wins += tile == 'G';
        REQUIRE(s.latest().reward == (tile == 'G' ? 1 : 0));
    }
    REQUIRE(falls > 500);
    REQUIRE(wins > 50);
}

TEST_CASE("the curve is the share won of the last 100 episodes", "[frozen]")
{
    q_session s(frozen_grid(2), 0.1, 0.95, 0.3, 2, 100);
    REQUIRE(lessons::live_curve(s) == std::vector<double>{});
    while (s.returns().size() < 350) {
        s.step();
    }
    REQUIRE(s.curve().size() == s.returns().size());
    for (std::size_t i = 0; i < s.curve().size(); i++) {
        const std::size_t first = i + 1 > 100 ? i + 1 - 100 : 0;
        double sum = 0;
        for (std::size_t j = first; j <= i; j++) {
            sum += s.returns()[j];
        }
        REQUIRE(s.curve()[i] == Approx(sum / static_cast<double>(i + 1 - first)));
    }
    int won = 0;
    for (double r : s.returns()) {
        won += r > 0;
    }
    REQUIRE(s.tally() == "won " + std::to_string(won) + " of 350");
    REQUIRE(s.curve_label() == "share won, of the last 100 episodes");
    REQUIRE(s.curve_now() == s.curve().back());
    s.step();
    REQUIRE(lessons::live_curve(s).back() == s.curve().back());
}

TEST_CASE("reset forgets the wins", "[frozen]")
{
    q_session s(frozen_grid(2), 0.1, 0.95, 0.3, 2, 100);
    while (s.returns().size() < 300) {
        s.step();
    }
    s.reset();
    REQUIRE(s.curve().empty());
    REQUIRE(s.tally() == "won 0 of 0");
    REQUIRE(s.curve_now() == 0);
}

TEST_CASE("the numbers shown are the library's, for every learner", "[frozen]")
{
    learns_the_formula<lessons::q_learning_rule<frozen_grid>>();
    learns_the_formula<lessons::sarsa_rule<frozen_grid>>();
    learns_the_formula<lessons::expected_sarsa_rule<frozen_grid>>();
}

TEST_CASE("a session plays what play_online plays", "[frozen]")
{
    using rule = lessons::q_learning_rule<frozen_grid>;
    using types = games::types<frozen_grid>;
    q_session s(frozen_grid(7), 0.1, 0.95, 0.1, 7, 100);
    frozen_grid game(7);
    std::mt19937 prng(7);
    types::policy memory;
    rule::learner_type learner;
    learner.alpha = 0.1;
    learner.gamma = 0.95;
    auto agent = [&](const position & p, const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<frozen_grid>(memory, p, legal, 0.1, prng);
    };
    std::vector<double> returns;
    for (int e = 0; e < 400; e++) {
        returns.push_back(games::play_online(game, agent, learner, memory, 100).total_reward);
    }
    while (s.returns().size() < 400) {
        s.step();
    }
    REQUIRE(s.returns() == returns);
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            for (unsigned int a = 0; a < 4; a++) {
                REQUIRE(s.q({x, y}, a) == memory.value(types::state(position{x, y}), types::action(a)));
            }
        }
    }
}

TEST_CASE("convergence: it learns to win far more than at random", "[frozen][convergence]")
{
    q_session s(frozen_grid(11), 0.1, 0.95, 0.1, 11, 100);
    while (s.returns().size() < 20000) {
        s.step();
    }
    const double wins = greedy_wins(s, 20000);
    INFO("greedy share won: " << wins);
    REQUIRE(wins > 0.55);
    REQUIRE(s.curve().back() > 0.3);
}

TEST_CASE("convergence: it steps away from the holes, as the best policy does", "[frozen][convergence]")
{
    q_session s(frozen_grid(11), 0.1, 0.95, 0.1, 11, 100);
    while (s.returns().size() < 20000) {
        s.step();
    }
    // the first cell of the second row has a hole to its right: choosing left, it can only go left, up or down, never right
    REQUIRE(s.best_action({0, 1}) == 3);
}
