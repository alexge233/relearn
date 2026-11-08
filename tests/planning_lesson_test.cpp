/**
 * Tests of the lessons' planning learner: that it is the library's Dyna-Q, one step at a time, that what it shows of
 * every imagined step is what the library really did, and that planning does what the book says.
 */
#include "../examples/games/dyna_maze.hpp"
#include "../examples/window/planning.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using rules = games::dyna_maze;
using types = games::types<rules>;
namespace lessons = games::lessons;
using dyna = lessons::dyna_session<rules>;

namespace
{
const double alpha = 0.1;
const double gamma_ = 0.95;
const double epsilon = 0.1;

/// @brief every value of @param a and @param b must be equal
template <class one_type, class two_type>
void same_values(const one_type & a,
                 const two_type & b)
{
    for (int y = 0; y < rules::height; y++) {
        for (int x = 0; x < rules::width; x++) {
            for (unsigned int m = 0; m < 4; m++) {
                REQUIRE(a.q({x, y}, m) == b.q({x, y}, m));
            }
        }
    }
}

/// @brief a session must learn exactly what the library's own loop does with n imagined steps
void same_as_the_library(unsigned int n,
                         unsigned int episodes)
{
    dyna s(rules(), alpha, gamma_, epsilon, n, 5);
    rules game;
    std::mt19937 prng(5);
    types::policy memory;
    relearn::dyna_q<types::state, types::action> learner(alpha, gamma_, n, 0);
    auto agent = [&](const games::position & p, const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<rules>(memory, p, legal, epsilon, prng);
    };
    std::vector<double> moves;
    for (unsigned int e = 0; e < episodes; e++) {
        moves.push_back(static_cast<double>(games::play_online(game, agent, learner, memory, 3000).steps));
    }
    while (s.curve().size() < episodes) {
        s.step();
    }
    // the real step that ended the last episode is followed by its imagined ones
    for (unsigned int i = 0; i < n; i++) {
        s.step();
    }
    REQUIRE(s.curve() == moves);
    for (int y = 0; y < rules::height; y++) {
        for (int x = 0; x < rules::width; x++) {
            for (unsigned int m = 0; m < 4; m++) {
                REQUIRE(s.q({x, y}, m) == memory.value(types::state(games::position{x, y}), types::action(m)));
            }
        }
    }
}

/// @return the moves of the first @param episodes episodes with @param n imagined steps
double moves_of(unsigned int n,
                unsigned int episodes,
                unsigned int seed)
{
    dyna s(rules(), alpha, gamma_, epsilon, n, seed);
    while (s.curve().size() < episodes) {
        s.step();
    }
    double sum = 0;
    for (double v : s.curve()) {
        sum += v;
    }
    return sum;
}
} // namespace

TEST_CASE("the session is the library's Dyna-Q, one step at a time", "[planning]")
{
    same_as_the_library(0, 15);
    same_as_the_library(5, 15);
    same_as_the_library(50, 15);
}

TEST_CASE("n = 0 is Q-learning", "[planning]")
{
    dyna s(rules(), alpha, gamma_, epsilon, 0, 9);
    lessons::tabular_session<rules, lessons::q_learning_rule<rules>> q(rules(), alpha, gamma_, epsilon, 9, 3000);
    for (int i = 0; i < 20000; i++) {
        s.step();
        q.step();
        REQUIRE(!s.imagining());
        REQUIRE(s.last_action() == q.last_action());
    }
    same_values(s, q);
    REQUIRE(s.name() == "Q-learning, n = 0");
}

TEST_CASE("a real step, then n imagined ones, in which the agent does not move", "[planning]")
{
    dyna s(rules(), alpha, gamma_, epsilon, 3, 2);
    for (int round = 0; round < 500; round++) {
        s.step();
        REQUIRE(!s.imagining());
        REQUIRE(s.imagined_count() == 0);
        const auto real_cell = s.at();
        const auto steps = s.steps();
        for (unsigned int i = 1; i <= 3; i++) {
            s.step();
            REQUIRE(s.imagining());
            REQUIRE(s.imagined_count() == i);
            REQUIRE(s.at() == real_cell);
            REQUIRE(s.steps() == steps);
        }
    }
    REQUIRE(s.real_steps() == 500);
    REQUIRE(s.imagined_steps() == 1500);
    REQUIRE(s.tally() == "imagined 1500");
    REQUIRE(s.name() == "Dyna-Q, n = 3");
}

TEST_CASE("what is shown of a step is what the library did, real or imagined", "[planning]")
{
    dyna s(rules(), 0.3, 0.8, 0.2, 10, 4);
    int imagined_endings = 0;
    int imagined = 0;
    for (int i = 0; i < 40000; i++) {
        s.step();
        const auto & n = s.latest();
        REQUIRE(n.valid);
        REQUIRE(n.alpha == 0.3);
        REQUIRE(n.gamma == 0.8);
        // the formula holds, against the value in the policy
        REQUIRE(n.learnt == Approx(n.believed + 0.3 * (n.reward + (n.ended ? 0.0 : 0.8 * n.next) - n.believed)).margin(1e-12));
        REQUIRE(s.q(s.from(), s.last_action()) == n.learnt);
        if (s.imagining()) {
            imagined++;
            REQUIRE(s.remembers(s.from()));
            // this world never changes, so memory is the truth: the game agrees with where it says the move led
            rules game;
            game.place(s.from());
            const auto out = game.step(s.last_action());
            REQUIRE(s.imagined_to() == out.next);
            REQUIRE(n.reward == out.reward);
            REQUIRE(n.ended == out.terminated);
            imagined_endings += n.ended;
            if (n.ended) {
                REQUIRE(n.next == 0);
            }
        }
    }
    REQUIRE(imagined > 30000);
    REQUIRE(imagined_endings > 10);
}

TEST_CASE("it only remembers the cells it has been in", "[planning]")
{
    dyna s(rules(), alpha, gamma_, epsilon, 5, 1);
    for (int y = 0; y < rules::height; y++) {
        for (int x = 0; x < rules::width; x++) {
            REQUIRE(!s.remembers({x, y}));
        }
    }
    s.step();
    REQUIRE(s.remembers(rules().start()));
    REQUIRE(!s.remembers(rules().goal()));
    for (int i = 0; i < 100; i++) {
        s.step();
    }
    rules maze;
    for (int y = 0; y < rules::height; y++) {
        for (int x = 0; x < rules::width; x++) {
            if (maze.wall({x, y})) {
                REQUIRE(!s.remembers({x, y}));
            }
        }
    }
}

TEST_CASE("the imagined steps of the last real step of an episode still come", "[planning]")
{
    dyna s(rules(), alpha, gamma_, epsilon, 4, 3);
    while (s.curve().empty()) {
        s.step();
    }
    REQUIRE(!s.playing());
    REQUIRE(s.at() == rules().goal());
    for (unsigned int i = 1; i <= 4; i++) {
        s.step();
        REQUIRE(s.imagined_count() == i);
        REQUIRE(s.at() == rules().goal());
        REQUIRE(!s.playing());
    }
    const auto episode = s.episode();
    s.step();
    REQUIRE(!s.imagining());
    REQUIRE(s.episode() == episode + 1);
}

TEST_CASE("the curve counts moves, and shows the episode under way", "[planning]")
{
    dyna s(rules(), alpha, gamma_, epsilon, 1, 3);
    REQUIRE(lessons::live_curve(s).empty());
    s.step();
    REQUIRE(lessons::live_curve(s) == std::vector<double>{1});
    s.step();
    REQUIRE(lessons::live_curve(s) == std::vector<double>{1});
    s.step();
    REQUIRE(lessons::live_curve(s) == std::vector<double>{2});
    while (s.curve().empty()) {
        s.step();
    }
    REQUIRE(s.curve()[0] == s.real_steps());
    REQUIRE(s.curve_label() == "moves to the treasure, per episode");
    REQUIRE(lessons::live_curve(s) == s.curve());
}

TEST_CASE("restart drops what was left to imagine, and keeps what was learnt", "[planning]")
{
    dyna s(rules(), alpha, gamma_, epsilon, 5, 3);
    for (int i = 0; i < 303; i++) {
        s.step();
    }
    REQUIRE(s.imagining());
    REQUIRE(s.imagined_count() == 2);
    const auto real = s.real_steps();
    const double kept = s.q(rules().start(), 1);
    s.restart();
    REQUIRE(!s.imagining());
    REQUIRE(s.imagined_count() == 0);
    REQUIRE(s.q(rules().start(), 1) == kept);
    s.step();
    REQUIRE(!s.imagining());
    REQUIRE(s.real_steps() == real + 1);
    REQUIRE(s.remembers(rules().start()));
}

TEST_CASE("reset forgets everything, its memory too, and plays the same again", "[planning]")
{
    dyna s(rules(), alpha, gamma_, epsilon, 5, 3);
    dyna fresh(rules(), alpha, gamma_, epsilon, 5, 3);
    while (s.curve().empty()) {
        s.step();
    }
    s.reset();
    REQUIRE(s.real_steps() == 0);
    REQUIRE(s.imagined_steps() == 0);
    REQUIRE(s.curve().empty());
    REQUIRE(!s.imagining());
    REQUIRE(!s.remembers(rules().start()));
    // a session that was never reset: the first move of its first step is the first move of the reset one
    for (int i = 0; i < 2000; i++) {
        s.step();
        fresh.step();
        // still told what it imagines after a reset
        REQUIRE(s.latest().learnt == fresh.latest().learnt);
        REQUIRE(s.from() == fresh.from());
    }
    same_values(s, fresh);
}

TEST_CASE("n changes from the next real step, and the lesson's value comes back", "[planning]")
{
    dyna s(rules(), alpha, gamma_, epsilon, 2, 3);
    s.step();
    s.set_planning_steps(1);
    s.step();
    s.step();
    REQUIRE(s.imagined_count() == 2);
    s.step();
    REQUIRE(!s.imagining());
    s.step();
    REQUIRE(s.imagining());
    s.step();
    REQUIRE(!s.imagining());
    REQUIRE(s.planning_steps() == 1);
    s.restore_settings();
    REQUIRE(s.planning_steps() == 2);
    s.set_alpha(0.9);
    s.restore_settings();
    REQUIRE(s.alpha() == alpha);
}

TEST_CASE("the sliders' alpha and gamma reach the imagined steps", "[planning]")
{
    dyna s(rules(), alpha, gamma_, epsilon, 4, 3);
    for (int i = 0; i < 3000; i++) {
        s.step();
    }
    s.set_alpha(0.7);
    s.set_gamma(0.5);
    for (int i = 0; i < 3000; i++) {
        s.step();
        const auto & n = s.latest();
        REQUIRE(n.alpha == 0.7);
        REQUIRE(n.learnt == Approx(n.believed + 0.7 * (n.reward + (n.ended ? 0.0 : 0.5 * n.next) - n.believed)).margin(1e-12));
    }
}

TEST_CASE("convergence: planning needs far fewer real moves, as in Figure 8.2", "[planning][convergence]")
{
    double none = 0;
    double some = 0;
    double many = 0;
    for (unsigned int seed = 1; seed <= 8; seed++) {
        none += moves_of(0, 10, seed);
        some += moves_of(5, 10, seed);
        many += moves_of(50, 10, seed);
    }
    REQUIRE(some < none / 2);
    REQUIRE(many < some);
}

TEST_CASE("convergence: it ends up on the shortest way, 14 moves", "[planning][convergence]")
{
    dyna s(rules(), alpha, gamma_, epsilon, 50, 3);
    while (s.curve().size() < 60) {
        s.step();
    }
    rules maze;
    games::position at = maze.reset();
    int moves = 0;
    bool over = false;
    while (!over && moves < 100) {
        const auto out = maze.step(s.best_action(at));
        at = out.next;
        over = out.terminated;
        moves++;
    }
    REQUIRE(over);
    REQUIRE(moves == 14);
}

TEST_CASE("the title of the formula says whether it was imagined", "[planning][words]")
{
    REQUIRE(lessons::formula_title(false, 0, 5) == "The whole idea, with the numbers from this step");
    REQUIRE(lessons::formula_title(true, 3, 5) == "The same formula, from memory: imagined step 3 of 5");
}

TEST_CASE("the words of an imagined step say where the facts came from", "[planning][words]")
{
    lessons::numbers n;
    n.believed = 0.5;
    n.alpha = 0.5;
    n.reward = 1;
    n.gamma = 0.5;
    n.next = 0.4;
    n.learnt = 0.85;
    n.valid = true;
    const auto lines = lessons::imagined_words(n, "up", "Best from there");
    REQUIRE(lines.size() == 4);
    REQUIRE(lines[0] == "It did not move. It picked something it remembers: going up from there, believed worth 0.50.");
    REQUIRE(lines[1] == "It remembers: reward 1.00. Best from there: 0.40.");
    REQUIRE(lines[2] == lessons::plain_words(n, "up", "Best from there")[2]);
    n.ended = true;
    REQUIRE(lessons::imagined_words(n, "up", "Best")[1] == "It remembers: reward 1.00, and the episode ended there.");
}

TEST_CASE("a whole number is written without decimals", "[planning][words]")
{
    REQUIRE(lessons::plain_number(1005) == "1005");
    REQUIRE(lessons::plain_number(-13) == "-13");
    REQUIRE(lessons::plain_number(0.5) == "0.50");
}
