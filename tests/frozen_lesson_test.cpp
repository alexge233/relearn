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
    // sliding left into the edge when it chose up
    REQUIRE(lessons::turn_of({0, 1}, 0, {0, 1}, 4, 4) == -1);
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
