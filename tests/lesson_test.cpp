/**
 * Tests of the lessons' teaching layer: what is written on the screen, worked out by hand, and above all that the
 * formula shown is the one the library really computed, at every step, for every learner.
 */
#include "../examples/games/cliff_walking.hpp"
#include "../examples/games/windy_gridworld.hpp"
#include "../examples/window/lesson.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using rules = games::cliff_walking;
namespace lessons = games::lessons;
using lessons::q_learning_rule;
using lessons::sarsa_rule;
using lessons::expected_sarsa_rule;

namespace
{
lessons::numbers sample()
{
    lessons::numbers n;
    n.believed = -26.03;
    n.alpha = 0.5;
    n.reward = -1;
    n.gamma = 1;
    n.next = -5.05;
    n.learnt = -16.04;
    n.valid = true;
    return n;
}

/// @brief play @param steps steps with @param rule, and require the update shown to be the one the library made
template <class rule>
void shown_is_what_was_learnt(unsigned int steps)
{
    lessons::tabular_session<rules, rule> s(rules(), 0.5, 0.9, 0.1, 11);
    unsigned int ended = 0;
    for (unsigned int i = 0; i < steps; i++) {
        s.step();
        const auto & n = s.latest();
        REQUIRE(n.valid);
        REQUIRE(n.learnt == Approx(n.believed + n.alpha * (n.reward + n.gamma * n.next - n.believed)).epsilon(1e-12).margin(1e-12));
        REQUIRE(n.learnt == Approx(n.believed + n.alpha * n.surprise()).epsilon(1e-12).margin(1e-12));
        if (n.ended) {
            ended++;
            REQUIRE(n.next == 0);
        }
    }
    REQUIRE(ended > 0);
}

/// @brief a session and games::play_online, from the same seed, must play and learn the very same thing
template <class rule>
void same_as_play_online(unsigned int episodes)
{
    using types = games::types<rules>;
    const double alpha = 0.5, gamma = 1.0, epsilon = 0.1;
    lessons::tabular_session<rules, rule> s(rules(), alpha, gamma, epsilon, 5);
    rules game;
    std::mt19937 prng(5);
    types::policy memory;
    typename rule::learner_type learner;
    learner.alpha = alpha;
    learner.gamma = gamma;
    rule::prepare(learner, memory, game, epsilon);
    auto agent = [&](const games::position & p, const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<rules>(memory, p, legal, epsilon, prng);
    };
    std::vector<double> returns;
    for (unsigned int e = 0; e < episodes; e++) {
        returns.push_back(games::play_online(game, agent, learner, memory, 1000).total_reward);
    }
    while (s.returns().size() < episodes) {
        s.step();
    }
    REQUIRE(s.returns() == returns);
    for (int y = 0; y < rules::height; y++) {
        for (int x = 0; x < rules::width; x++) {
            for (unsigned int a = 0; a < 4; a++) {
                REQUIRE(s.q({x, y}, a) == memory.value(types::state(games::position{x, y}), types::action(a)));
            }
        }
    }
}
} // namespace

TEST_CASE("two decimals, and never minus zero", "[lesson][words]")
{
    REQUIRE(lessons::two_decimals(1.234) == "1.23");
    REQUIRE(lessons::two_decimals(-16.039) == "-16.04");
    REQUIRE(lessons::two_decimals(0) == "0.00");
    REQUIRE(lessons::two_decimals(-0.001) == "0.00");
    REQUIRE(lessons::two_decimals(0.001) == "0.00");
    REQUIRE(lessons::two_decimals(-0.006) == "-0.01");
}

TEST_CASE("numbers: the target and the surprise", "[lesson][numbers]")
{
    lessons::numbers n;
    n.believed = -26;
    n.reward = -1;
    n.gamma = 0.5;
    n.next = -4;
    REQUIRE(n.target() == -3);
    REQUIRE(n.surprise() == 23);
    REQUIRE(!n.valid);
    REQUIRE(!n.ended);
}

TEST_CASE("the formula: its columns, numbers and meanings", "[lesson][formula]")
{
    const auto terms = lessons::formula_terms(sample(), "max Q(s',a')", "best from there");
    REQUIRE(terms.size() == 13);
    const std::vector<std::string> symbols = {"Q(s,a)", "arrow", "Q(s,a)", "+", "alpha", "[", "r", "+", "gamma", "max Q(s',a')", "-", "Q(s,a)", "]"};
    const std::vector<std::string> values  = {"-16.04", "=", "-26.03", "+", "0.50", "[", "-1.00", "+", "1.00", "-5.05", "-", "-26.03", "]"};
    const std::vector<std::string> meaning = {"new belief", "", "I believed", "", "speed", "", "reward", "", "future", "best from there", "", "I believed", ""};
    for (std::size_t i = 0; i < terms.size(); i++) {
        REQUIRE(terms[i].symbol == symbols[i]);
        REQUIRE(terms[i].number == values[i]);
        REQUIRE(terms[i].label == meaning[i]);
    }
    REQUIRE(terms[0].kind == lessons::role::result);
    REQUIRE(terms[2].kind == lessons::role::belief);
    REQUIRE(terms[4].kind == lessons::role::speed);
    REQUIRE(terms[6].kind == lessons::role::reward);
    REQUIRE(terms[8].kind == lessons::role::future);
    REQUIRE(terms[9].kind == lessons::role::next);
    REQUIRE(terms[11].kind == lessons::role::belief);
    REQUIRE(terms[1].kind == lessons::role::plain);
}

TEST_CASE("the formula's columns: centred, and squeezed to fit", "[lesson][formula]")
{
    // 2 columns of 100 and 50, with a gap of 10: 170 wide, centred in 370 from 30: starts at 130
    auto c = lessons::formula_columns({100, 50}, 30, 370, 10);
    REQUIRE(c[0].x == 130);
    REQUIRE(c[0].width == 110);
    REQUIRE(c[1].x == 240);
    REQUIRE(c[1].width == 60);
    // 150 of content in 170: the gap of 40 becomes (170 - 150) / 2 = 10
    c = lessons::formula_columns({100, 50}, 0, 170, 40);
    REQUIRE(c[0].width == 110);
    REQUIRE(c[1].width == 60);
    REQUIRE(c[1].x + c[1].width == 170);
    // no room at all: no gap, never a negative one
    c = lessons::formula_columns({100, 50}, 0, 100, 40);
    REQUIRE(c[0].width == 100);
    REQUIRE(c[1].width == 50);
}

TEST_CASE("the plain words", "[lesson][words]")
{
    auto lines = lessons::plain_words(sample(), "down", "Best it can do from the next cell");
    REQUIRE(lines.size() == 4);
    REQUIRE(lines[0] == "It believed going down from here was worth -26.03.");
    REQUIRE(lines[1] == "It went. Reward -1.00. Best it can do from the next cell: -5.05.");
    REQUIRE(lines[2] == "So the move was really worth -6.05. Surprise: 19.98.");
    REQUIRE(lines[3] == "It moves its belief 50% of the way there: now -16.04.");
    // the episode ended: nothing comes after
    auto ended = sample();
    ended.ended = true;
    ended.next = 0;
    REQUIRE(lessons::plain_words(ended, "right", "x")[1] == "It went. Reward -1.00. The episode is over, so nothing comes after.");
    // no surprise
    lessons::numbers right;
    right.believed = -3;
    right.reward = -1;
    right.gamma = 1;
    right.next = -2;
    right.alpha = 0.1;
    right.learnt = -3;
    REQUIRE(lessons::plain_words(right, "up", "x")[2] == "So the move was really worth -3.00. No surprise: it was right.");
    REQUIRE(lessons::plain_words(right, "up", "x")[3] == "It moves its belief 10% of the way there: now -3.00.");
}

TEST_CASE("the update shown is the one the library made, at every step", "[lesson][validation]")
{
    shown_is_what_was_learnt<q_learning_rule<rules>>(4000);
    shown_is_what_was_learnt<sarsa_rule<rules>>(4000);
    shown_is_what_was_learnt<expected_sarsa_rule<rules>>(4000);
}

TEST_CASE("a session plays and learns exactly as games::play_online", "[lesson][validation]")
{
    same_as_play_online<q_learning_rule<rules>>(80);
    same_as_play_online<sarsa_rule<rules>>(80);
    same_as_play_online<expected_sarsa_rule<rules>>(80);
}

TEST_CASE("what comes next, for each rule, worked out by hand", "[lesson][rules]")
{
    using types = games::types<rules>;
    types::policy m;
    const games::position p{2, 1};
    m.update(types::state(p), types::action(0u), -2);
    m.update(types::state(p), types::action(1u), -6);
    const std::vector<unsigned int> legal = {0, 1, 2, 3};
    // only stored actions count for the best: -2, not the 0 of the untried ones
    REQUIRE((q_learning_rule<rules>::next_value(m, p, 1, legal, 0.1)) == -2);
    // an unseen cell is worth 0
    REQUIRE((q_learning_rule<rules>::next_value(m, games::position{9, 0}, 1, legal, 0.1)) == 0);
    // SARSA: the move chosen, whatever it is
    REQUIRE((sarsa_rule<rules>::next_value(m, p, 1, legal, 0.1)) == -6);
    REQUIRE((sarsa_rule<rules>::next_value(m, p, 0, legal, 0.1)) == -2);
    // Expected SARSA: actions 2 and 3, untried, are the best (0): each 0.1 / 4 + 0.9 / 2; the others 0.1 / 4
    REQUIRE((expected_sarsa_rule<rules>::next_value(m, p, 0, legal, 0.1)) == Approx(0.025 * -2 + 0.025 * -6));
    REQUIRE(std::string(q_learning_rule<rules>::name()) == "Q-learning");
    REQUIRE(std::string(sarsa_rule<rules>::next_symbol()) == "Q(s',a')");
    REQUIRE(std::string(expected_sarsa_rule<rules>::next_symbol()) == "E[Q(s',a')]");
}

TEST_CASE("a session: its first step, the cliff, and what it reports", "[lesson][session]")
{
    lessons::tabular_session<rules, q_learning_rule<rules>> s(rules(), 0.5, 1.0, 0.0, 1);
    REQUIRE(!s.latest().valid);
    REQUIRE(s.episode() == 0);
    REQUIRE(s.name() == "Q-learning");
    REQUIRE(s.epsilon() == 0);
    REQUIRE(s.moves().size() == 4);
    s.step();
    REQUIRE(s.episode() == 1);
    REQUIRE(s.steps() == 1);
    REQUIRE(s.latest().valid);
    REQUIRE(s.from() == rules::start());
    REQUIRE(s.known(rules::start()));
    REQUIRE(!s.known({9, 0}));
    REQUIRE(s.q({9, 0}, 0u) == 0);
    REQUIRE(s.total() == s.latest().reward);
    // walk until it falls: the fall is punished, and sends it back to the start
    lessons::tabular_session<rules, sarsa_rule<rules>> faller(rules(), 0.5, 1.0, 1.0, 3);
    bool fell = false;
    for (int i = 0; i < 5000 && !fell; i++) {
        faller.step();
        fell = faller.punished();
    }
    REQUIRE(fell);
    REQUIRE(faller.latest().reward == -100);
    REQUIRE(faller.at() == rules::start());
    faller.step();
    REQUIRE(!faller.punished());
}

TEST_CASE("a session: the best action is the first of the best, as the agent's", "[lesson][session]")
{
    lessons::tabular_session<rules, q_learning_rule<rules>> s(rules(), 1.0, 1.0, 0.0, 1);
    const games::position untouched{6, 0};
    REQUIRE(s.best_action(untouched) == 0);
    REQUIRE(s.best(untouched) == 0);
    // from the start the agent, learning with alpha = 1, knows up after its first update
    for (int i = 0; i < 400; i++) {
        s.step();
    }
    const games::position start = rules::start();
    double top = s.q(start, 0u);
    for (unsigned int a = 1; a < 4; a++) {
        top = std::max(top, s.q(start, a));
    }
    REQUIRE(s.best(start) == top);
    REQUIRE(s.q(start, s.best_action(start)) == top);
}

TEST_CASE("a session: a cut-short episode, restart and reset", "[lesson][session]")
{
    using session = lessons::tabular_session<rules, q_learning_rule<rules>>;
    // cut short after 5 steps of random walking: its return is recorded
    session cut(rules(), 0.5, 1.0, 1.0, 2, 5);
    for (int i = 0; i < 5; i++) {
        cut.step();
    }
    REQUIRE(cut.returns().size() == 1);
    REQUIRE(cut.returns()[0] == cut.total());
    cut.step();
    REQUIRE(cut.episode() == 2);
    REQUIRE(cut.steps() == 1);

    session s(rules(), 0.5, 1.0, 0.1, 9);
    for (int i = 0; i < 40; i++) {
        s.step();
    }
    const double learnt = s.q(rules::start(), 0u);
    const auto episode = s.episode();
    const auto returns = s.returns().size();
    // restart: a new episode, keeping what was learnt, the abandoned one not counted
    s.restart();
    REQUIRE(s.episode() == episode + 1);
    REQUIRE(s.steps() == 0);
    REQUIRE(s.total() == 0);
    REQUIRE(s.at() == rules::start());
    REQUIRE(s.returns().size() == returns);
    REQUIRE(!s.latest().valid);
    REQUIRE(s.q(rules::start(), 0u) == learnt);
    // reset: everything forgotten
    for (int i = 0; i < 3000; i++) {
        s.step();
    }
    REQUIRE(!s.returns().empty());
    s.reset();
    REQUIRE(s.returns().empty());
    REQUIRE(s.episode() == 1);
    REQUIRE(s.steps() == 0);
    REQUIRE(!s.latest().valid);
    for (int y = 0; y < rules::height; y++) {
        for (int x = 0; x < rules::width; x++) {
            REQUIRE(!s.known({x, y}));
        }
    }
    // and it then plays as a new one does, drawing the same numbers (the first move has been chosen in both)
    session fresh(rules(), 0.5, 1.0, 0.1, 9);
    fresh.restart();
    for (int i = 0; i < 1500; i++) {
        s.step();
        fresh.step();
        REQUIRE(s.at() == fresh.at());
        REQUIRE(s.latest().learnt == fresh.latest().learnt);
    }
}

TEST_CASE("a session works on another grid game", "[lesson][session]")
{
    lessons::tabular_session<games::windy_gridworld, sarsa_rule<games::windy_gridworld>> s(games::windy_gridworld(), 0.5, 1.0, 0.1, 4);
    for (int i = 0; i < 2000; i++) {
        s.step();
        const auto & n = s.latest();
        REQUIRE(n.learnt == Approx(n.believed + n.alpha * n.surprise()).epsilon(1e-12).margin(1e-12));
    }
    REQUIRE(!s.returns().empty());
}

TEST_CASE("controls: the speed", "[lesson][controls]")
{
    lessons::controls c;
    REQUIRE(c.rate() == 5);
    REQUIRE(c.gliding());
    c.slower();
    REQUIRE(c.rate() == 2);
    c.slower();
    c.slower();
    c.slower();
    REQUIRE(c.rate() == 1);
    for (int i = 0; i < 20; i++) {
        c.faster();
    }
    REQUIRE(c.rate() == 3000);
    REQUIRE(!c.gliding());
    lessons::controls edge;
    edge.faster();
    REQUIRE(edge.rate() == 10);
    REQUIRE(edge.gliding());
    edge.faster();
    REQUIRE(edge.rate() == 30);
    REQUIRE(!edge.gliding());
    REQUIRE(c.values);
    REQUIRE(c.arrows);
    REQUIRE(!c.help);
    REQUIRE(!c.paused);
}

TEST_CASE("controls: the steps of a frame", "[lesson][controls]")
{
    lessons::controls c;
    // 5 steps a second: 0.1 s is half a step, carried over to the next frame
    REQUIRE(c.steps_for(0.1) == 0);
    REQUIRE(c.progress() == Approx(0.5));
    REQUIRE(c.steps_for(0.1) == 1);
    REQUIRE(c.progress() == Approx(0).margin(1e-9));
    REQUIRE(c.steps_for(1.0) == 5);
    // paused: none, however long
    c.paused = true;
    REQUIRE(c.steps_for(10) == 0);
    REQUIRE(c.progress() == 1);
    // a single step, asked for twice, is played twice, and only once
    c.single_step();
    c.single_step();
    REQUIRE(c.steps_for(0.016) == 2);
    REQUIRE(c.steps_for(0.016) == 0);
    // asking for one step pauses
    lessons::controls running;
    running.single_step();
    REQUIRE(running.paused);
    REQUIRE(running.steps_for(0.016) == 1);
    // a step asked for whilst paused is not left over for after it
    running.single_step();
    running.paused = false;
    REQUIRE(running.steps_for(0.0) == 0);
    running.paused = true;
    REQUIRE(running.steps_for(0.0) == 0);
    // too fast to glide: the agent is simply there
    lessons::controls fast;
    for (int i = 0; i < 3; i++) {
        fast.faster();
    }
    fast.steps_for(0.0001);
    REQUIRE(fast.progress() == 1);
    // progress never goes past 1
    lessons::controls slow;
    slow.steps_for(0.05);
    REQUIRE(slow.progress() <= 1);
}

TEST_CASE("the keys", "[lesson][controls]")
{
    using lessons::command;
    REQUIRE(lessons::command_of_key(' ') == command::pause);
    REQUIRE(lessons::command_of_key('N') == command::step);
    REQUIRE(lessons::command_of_key(262) == command::step);
    REQUIRE(lessons::command_of_key('R') == command::restart);
    REQUIRE(lessons::command_of_key('Z') == command::reset);
    REQUIRE(lessons::command_of_key('-') == command::slower);
    REQUIRE(lessons::command_of_key(333) == command::slower);
    REQUIRE(lessons::command_of_key('=') == command::faster);
    REQUIRE(lessons::command_of_key(334) == command::faster);
    REQUIRE(lessons::command_of_key('V') == command::values);
    REQUIRE(lessons::command_of_key('A') == command::arrows);
    REQUIRE(lessons::command_of_key('I') == command::intro);
    REQUIRE(lessons::command_of_key('H') == command::help);
    REQUIRE(lessons::command_of_key('Q') == command::quit);
    REQUIRE(lessons::command_of_key(258) == command::learner);
    REQUIRE(lessons::command_of_key('X') == command::none);
}

TEST_CASE("the top bar: where the entries are, and what a click hits", "[lesson][menu]")
{
    using lessons::command;
    const std::vector<std::pair<command, std::string>> entries = {{command::restart, "aaaa"}, {command::reset, "bb"}};
    // each character is 10 wide, with 12 each side: 64, then 44, from 8
    const auto items = lessons::lay_out_menu(entries, [](const std::string & s) { return 10.0f * s.size(); }, 8, 12);
    REQUIRE(items.size() == 2);
    REQUIRE(items[0].x == 8);
    REQUIRE(items[0].width == 64);
    REQUIRE(items[1].x == 72);
    REQUIRE(items[1].width == 44);
    REQUIRE(items[1].label == "bb");
    REQUIRE(lessons::menu_hit(items, 8, 5, 36) == command::restart);
    REQUIRE(lessons::menu_hit(items, 71.5f, 5, 36) == command::restart);
    REQUIRE(lessons::menu_hit(items, 72, 5, 36) == command::reset);
    REQUIRE(lessons::menu_hit(items, 115.5f, 5, 36) == command::reset);
    REQUIRE(lessons::menu_hit(items, 116, 5, 36) == command::none);
    REQUIRE(lessons::menu_hit(items, 7, 5, 36) == command::none);
    REQUIRE(lessons::menu_hit(items, 20, 36, 36) == command::none);
    REQUIRE(lessons::menu_hit(items, 20, 35.5f, 36) == command::restart);
    REQUIRE(lessons::menu_hit(items, 20, -1, 36) == command::none);
}

TEST_CASE("wrapping text", "[lesson][words]")
{
    REQUIRE(lessons::wrap_text("one two three", 7) == std::vector<std::string>{"one two", "three"});
    REQUIRE(lessons::wrap_text("one two three", 3) == std::vector<std::string>{"one", "two", "three"});
    REQUIRE(lessons::wrap_text("one two three", 80) == std::vector<std::string>{"one two three"});
    REQUIRE(lessons::wrap_text("", 10).empty());
    REQUIRE(lessons::wrap_text("abcd efg", 8) == std::vector<std::string>{"abcd efg"});
    REQUIRE(lessons::wrap_text("abcd efgh", 8) == std::vector<std::string>{"abcd", "efgh"});
}

TEST_CASE("the colours of the values, and the scale", "[lesson][colours]")
{
    REQUIRE(lessons::heat_colour(0) == games::rgb{30, 50, 170});
    REQUIRE(lessons::heat_colour(1) == games::rgb{255, 225, 90});
    REQUIRE(lessons::heat_colour(1.0 / 3) == games::rgb{140, 60, 170});
    REQUIRE(lessons::heat_colour(2.0 / 3) == games::rgb{240, 120, 60});
    REQUIRE(lessons::heat_colour(-1) == lessons::heat_colour(0));
    REQUIRE(lessons::heat_colour(9) == lessons::heat_colour(1));
    // half way between the first two stops
    REQUIRE(lessons::heat_colour(1.0 / 6) == games::gradient(0.5, games::rgb{30, 50, 170}, games::rgb{140, 60, 170}));
    REQUIRE(lessons::scaled(5, 0, 10) == 0.5);
    REQUIRE(lessons::scaled(-3, -3, 7) == 0);
    REQUIRE(lessons::scaled(4, 4, 4) == 1);
    REQUIRE(lessons::lowest_shown({-10, -700, 3}, -300) == -300);
    REQUIRE(lessons::lowest_shown({-10, -70, 3}, -300) == -70);
    REQUIRE(lessons::lowest_shown({}, -300) == 0);
}

TEST_CASE("the points of a learning curve", "[lesson][colours]")
{
    // few values are left alone
    REQUIRE(lessons::curve_points({1, 2, 3}, 3) == std::vector<double>{1, 2, 3});
    REQUIRE(lessons::curve_points({1, 2, 3}, 0) == std::vector<double>{1, 2, 3});
    REQUIRE(lessons::curve_points({}, 5).empty());
    // 4 values into 2 columns are the means of each half
    REQUIRE(lessons::curve_points({-5, 5, 8, 12}, 2) == std::vector<double>{0, 10});
    // 5 values into 3 columns: 2, 2 and 1 of them
    REQUIRE(lessons::curve_points({1, 3, 5, 7, 9}, 3) == std::vector<double>{2, 6, 9});
    // never more points than columns, whatever the number of episodes
    REQUIRE(lessons::curve_points(std::vector<double>(100000, -13), 136).size() == 136);
    REQUIRE(lessons::curve_points(std::vector<double>(100000, -13), 136)[135] == Approx(-13));
}

TEST_CASE("hot zones: which one is under the mouse", "[lesson][bubbles]")
{
    const std::vector<lessons::hot_zone> zones = {{10, 10, 100, 50, "big"}, {20, 20, 10, 10, "small"}};
    REQUIRE(lessons::zone_at(zones, 5, 15) == nullptr);
    REQUIRE(lessons::zone_at(zones, 10, 10)->text == "big");
    REQUIRE(lessons::zone_at(zones, 109.5f, 59.5f)->text == "big");
    REQUIRE(lessons::zone_at(zones, 110, 30) == nullptr);
    REQUIRE(lessons::zone_at(zones, 50, 60) == nullptr);
    // the later one is on top, and only inside its own bounds
    REQUIRE(lessons::zone_at(zones, 25, 25)->text == "small");
    REQUIRE(lessons::zone_at(zones, 30, 25)->text == "big");
    REQUIRE(lessons::zone_at({}, 1, 1) == nullptr);
}

TEST_CASE("where a bubble goes", "[lesson][bubbles]")
{
    // room everywhere: below and to the right of the mouse
    auto p = lessons::place_bubble(100, 100, 200, 60, 1280, 800);
    REQUIRE(p.x == 116);
    REQUIRE(p.y == 116);
    // at the right edge: to its left; at the bottom: above it
    p = lessons::place_bubble(1200, 100, 200, 60, 1280, 800);
    REQUIRE(p.x == 984);
    REQUIRE(p.y == 116);
    p = lessons::place_bubble(100, 780, 200, 60, 1280, 800);
    REQUIRE(p.x == 116);
    REQUIRE(p.y == 704);
    // exactly fitting stays on the usual side: 1280 - 6 - 200 - 16 = 1058
    REQUIRE(lessons::place_bubble(1058, 100, 200, 60, 1280, 800).x == 1074);
    REQUIRE(lessons::place_bubble(1059, 100, 200, 60, 1280, 800).x == 843);
    // never outside the window, even with no good side
    p = lessons::place_bubble(50, 50, 1300, 900, 1280, 800);
    REQUIRE(p.x == 6);
    REQUIRE(p.y == 6);
    p = lessons::place_bubble(10, 790, 300, 790, 1280, 800);
    REQUIRE(p.y >= 6);
    REQUIRE(p.y + 790 <= 800);
}

TEST_CASE("a bubble waits for the mouse to rest", "[lesson][bubbles]")
{
    lessons::hover_timer h;
    // the first frame on a zone only starts the wait
    REQUIRE(!h.update("a", 1.0));
    REQUIRE(!h.update("a", 0.2));
    REQUIRE(h.update("a", 0.1));
    REQUIRE(h.update("a", 5));
    // on another zone it waits again
    REQUIRE(!h.update("b", 5));
    REQUIRE(h.update("b", 0.3));
    // nothing under the mouse: no bubble, however long
    REQUIRE(!h.update("", 0.1));
    REQUIRE(!h.update("", 9));
    // and back on the same zone it waits again
    REQUIRE(!h.update("b", 0.1));
}

TEST_CASE("what the bubbles say", "[lesson][bubbles]")
{
    using lessons::command;
    for (auto c : {command::learner, command::restart, command::reset, command::pause, command::step, command::slower,
                   command::faster, command::values, command::arrows, command::intro, command::help}) {
        REQUIRE(!lessons::menu_help(c).empty());
    }
    REQUIRE(lessons::menu_help(command::none).empty());
    REQUIRE(lessons::menu_help(command::reset).find("(Z)") != std::string::npos);
    REQUIRE(lessons::menu_help(command::restart).find("(R)") != std::string::npos);
    // every term that has a meaning explains it, and the plain signs that do not stay silent
    const auto terms = lessons::formula_terms(sample(), "max Q(s',a')", "best from there");
    std::vector<std::string> said;
    for (const auto & t : terms) {
        said.push_back(lessons::term_help(t, "THE IDEA"));
    }
    REQUIRE(said[0].find("new belief") != std::string::npos);
    REQUIRE(said[1].find("Replace") != std::string::npos);
    REQUIRE(said[2].find("old belief") != std::string::npos);
    REQUIRE(said[3].empty());
    REQUIRE(said[4].find("alpha") != std::string::npos);
    REQUIRE(said[5].find("surprise") != std::string::npos);
    REQUIRE(said[6].find("reward") != std::string::npos);
    REQUIRE(said[7].empty());
    REQUIRE(said[8].find("gamma") != std::string::npos);
    REQUIRE(said[9].find("THE IDEA") != std::string::npos);
    REQUIRE(said[10].empty());
    REQUIRE(said[12] == said[5]);
    // a cell: unvisited, then what it believes
    lessons::tabular_session<rules, q_learning_rule<rules>> s(rules(), 1.0, 1.0, 0.0, 1);
    const std::vector<std::string> names = {"up", "right", "down", "left"};
    REQUIRE(lessons::cell_help(s, {9, 0}, names) == "It has not been here yet, so it knows nothing about this cell.");
    for (int i = 0; i < 60; i++) {
        s.step();
    }
    const games::position start = rules::start();
    REQUIRE(lessons::cell_help(s, start, names) == "Its best known move here is " + names[s.best_action(start)] + ", which it believes is worth " + lessons::two_decimals(s.best(start)) + ".");
}

TEST_CASE("the learning curve moves from the first step", "[lesson][curve]")
{
    lessons::tabular_session<rules, q_learning_rule<rules>> s(rules(), 0.5, 1.0, 0.1, 6);
    REQUIRE(lessons::live_curve(s).empty());
    s.step();
    REQUIRE(lessons::live_curve(s) == std::vector<double>{s.total()});
    s.step();
    REQUIRE(lessons::live_curve(s) == std::vector<double>{s.total()});
    // an episode that has ended is on the curve once, not twice
    while (s.returns().empty()) {
        s.step();
    }
    REQUIRE(!s.playing());
    REQUIRE(lessons::live_curve(s) == s.returns());
    s.step();
    REQUIRE(s.playing());
    REQUIRE(lessons::live_curve(s).size() == 2);
    REQUIRE(lessons::live_curve(s)[0] == s.returns()[0]);
    REQUIRE(lessons::live_curve(s)[1] == s.total());
}

TEST_CASE("the settings can change whilst it learns, and the formula stays true", "[lesson][settings]")
{
    lessons::tabular_session<rules, q_learning_rule<rules>> q(rules(), 0.5, 1.0, 0.1, 6);
    lessons::tabular_session<rules, sarsa_rule<rules>> sarsa(rules(), 0.5, 1.0, 0.1, 6);
    lessons::tabular_session<rules, expected_sarsa_rule<rules>> expected(rules(), 0.5, 1.0, 0.1, 6);
    lessons::session * all[] = {&q, &sarsa, &expected};
    for (auto * s : all) {
        REQUIRE(s->alpha() == 0.5);
        REQUIRE(s->gamma() == 1.0);
        REQUIRE(s->epsilon() == 0.1);
        const double settings[][3] = {{0.5, 1.0, 0.1}, {0.1, 0.9, 0.3}, {1.0, 0.5, 0.0}, {0.25, 0.0, 1.0}};
        for (const auto & setting : settings) {
            s->set_alpha(setting[0]);
            s->set_gamma(setting[1]);
            s->set_epsilon(setting[2]);
            for (int i = 0; i < 400; i++) {
                s->step();
                const auto & n = s->latest();
                REQUIRE(n.alpha == setting[0]);
                REQUIRE(n.gamma == setting[1]);
                REQUIRE(n.learnt == Approx(n.believed + n.alpha * n.surprise()).epsilon(1e-12).margin(1e-12));
            }
        }
        // kept between 0 and 1
        s->set_alpha(7);
        REQUIRE(s->alpha() == 1);
        s->set_alpha(-1);
        REQUIRE(s->alpha() == 0);
        s->set_gamma(2);
        REQUIRE(s->gamma() == 1);
        s->set_gamma(-2);
        REQUIRE(s->gamma() == 0);
        s->set_epsilon(3);
        REQUIRE(s->epsilon() == 1);
        s->set_epsilon(-3);
        REQUIRE(s->epsilon() == 0);
        // and back to the lesson's values
        s->restore_settings();
        REQUIRE(s->alpha() == 0.5);
        REQUIRE(s->gamma() == 1.0);
        REQUIRE(s->epsilon() == 0.1);
        s->reset();
        REQUIRE(s->alpha() == 0.5);
    }
    // a reset keeps the settings: it forgets what was learnt, not how it learns
    q.set_alpha(0.2);
    q.reset();
    REQUIRE(q.alpha() == 0.2);
    q.step();
    REQUIRE(q.latest().alpha == 0.2);
}

TEST_CASE("epsilon really changes what the agent does", "[lesson][settings]")
{
    // with epsilon 1 it moves at random, and falls again and again
    lessons::tabular_session<rules, sarsa_rule<rules>> s(rules(), 0.5, 1.0, 0.1, 8);
    s.set_epsilon(1.0);
    int falls = 0;
    for (int i = 0; i < 3000; i++) {
        s.step();
        falls += s.punished();
    }
    REQUIRE(falls > 50);
    // a learnt policy followed without exploring never falls
    lessons::tabular_session<rules, sarsa_rule<rules>> greedy(rules(), 0.5, 1.0, 0.1, 8);
    for (int i = 0; i < 40000; i++) {
        greedy.step();
    }
    greedy.set_epsilon(0.0);
    int late = 0;
    for (int i = 0; i < 2000; i++) {
        greedy.step();
        late += greedy.punished();
    }
    REQUIRE(late == 0);
}

TEST_CASE("sliders: from the mouse to a value and back", "[lesson][settings]")
{
    const lessons::slider_range range{0, 1, 0.01};
    REQUIRE(lessons::slider_value(range, 100, 100, 200) == 0);
    REQUIRE(lessons::slider_value(range, 300, 100, 200) == 1);
    REQUIRE(lessons::slider_value(range, 200, 100, 200) == Approx(0.5));
    REQUIRE(lessons::slider_value(range, 150, 100, 200) == Approx(0.25));
    // beyond the ends of the track
    REQUIRE(lessons::slider_value(range, 50, 100, 200) == 0);
    REQUIRE(lessons::slider_value(range, 900, 100, 200) == 1);
    // on the steps: 0.2537 goes to 0.25
    REQUIRE(lessons::slider_value({0, 1, 0.05}, 150.74f, 100, 200) == Approx(0.25));
    REQUIRE(lessons::slider_value({0, 1, 0.05}, 156, 100, 200) == Approx(0.3));
    // a range not starting at 0
    REQUIRE(lessons::slider_value({0.5, 2.5, 0.5}, 150, 100, 200) == Approx(1.0));
    REQUIRE(lessons::slider_fraction(range, 0.3) == Approx(0.3));
    REQUIRE(lessons::slider_fraction({0.5, 2.5, 0.5}, 1.5) == Approx(0.5));
    REQUIRE(lessons::slider_fraction(range, 2) == 1);
    REQUIRE(lessons::slider_fraction(range, -2) == 0);
}

TEST_CASE("the drawn symbols", "[lesson][glyphs]")
{
    for (const char * name : {"alpha", "gamma", "epsilon", "arrow"}) {
        const auto rows = lessons::glyph(name);
        REQUIRE(!rows.empty());
        std::size_t set = 0;
        for (const auto & row : rows) {
            REQUIRE(row.size() == rows.front().size());
            for (char c : row) {
                REQUIRE((c == '#' || c == '.'));
                set += c == '#';
            }
        }
        REQUIRE(set > 10);
    }
    REQUIRE(lessons::glyph("Q(s,a)").empty());
    // gamma comes below the line, as a gamma does, unlike alpha: it is what tells it from a v
    REQUIRE(lessons::glyph("gamma").size() > lessons::glyph("alpha").size());
    REQUIRE(lessons::glyph("gamma").back().find('#') != std::string::npos);
}

TEST_CASE("the crawl, and the command line", "[lesson][intro]")
{
    REQUIRE(lessons::crawl_position(0, 46) == 0);
    REQUIRE(lessons::crawl_position(2, 46) == 92);
    REQUIRE(lessons::crawl_position(-1, 46) == 0);
    const char * none[] = {"p"};
    auto o = lessons::lesson_options::parse(1, const_cast<char **>(none));
    REQUIRE(!o.no_intro);
    REQUIRE(o.capture.empty());
    const char * all[] = {"p", "--no-intro", "--capture", "dir", "--other"};
    o = lessons::lesson_options::parse(5, const_cast<char **>(all));
    REQUIRE(o.no_intro);
    REQUIRE(o.capture == "dir");
    const char * dangling[] = {"p", "--capture"};
    REQUIRE(lessons::lesson_options::parse(2, const_cast<char **>(dangling)).capture.empty());
}
