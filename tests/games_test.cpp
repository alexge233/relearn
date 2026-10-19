/**
 * Tests of the example games: their rules, and that relearn's learners reach
 * the known answer for each one (exact solutions from `solvers.hpp`, or the
 * optimal path lengths given by Sutton & Barto).
 * Random number generators are seeded, so results are reproducible.
 */
#include "../examples/games/blackjack.hpp"
#include "../examples/games/cliff_walking.hpp"
#include "../examples/games/windy_gridworld.hpp"
#include "../examples/games/frozen_lake.hpp"
#include "solvers.hpp"
#include <cmath>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

namespace
{
/// train @param memory on @param episodes episodes of @param game, exploring with ε-greedy
template <class game_type, class learner_type>
void train(game_type & game,
           learner_type & learner,
           typename games::types<game_type>::policy & memory,
           unsigned int episodes,
           double epsilon,
           std::size_t max_steps,
           std::mt19937 & prng)
{
    for (unsigned int i = 0; i < episodes; i++) {
        auto r = games::play(game,
                             [&](const typename game_type::state_type & s,
                                 const std::vector<typename game_type::action_type> & legal) {
                                 return games::epsilon_greedy<game_type>(memory, s, legal, epsilon, prng);
                             },
                             max_steps);
        learner(r.episode, memory, r.terminated);
    }
}
} // namespace

// ------------------------------------------------------------------ blackjack

TEST_CASE("blackjack: hand totals", "[games][blackjack]")
{
    games::blackjack_hand two_aces;
    two_aces.add(1);
    two_aces.add(1);
    REQUIRE(two_aces.sum == 12);        // one Ace counts 11, the other 1
    REQUIRE(two_aces.usable_ace);

    games::blackjack_hand natural;
    natural.add(1);
    natural.add(10);
    REQUIRE(natural.natural());

    games::blackjack_hand soft;         // A + 5 + 9: the Ace must count 1
    soft.add(1);
    soft.add(5);
    REQUIRE(soft.sum == 16);
    REQUIRE(soft.usable_ace);
    soft.add(9);
    REQUIRE(soft.sum == 15);
    REQUIRE(!soft.usable_ace);
    REQUIRE(!soft.bust());
    REQUIRE(!soft.natural());
}

TEST_CASE("blackjack: game flow", "[games][blackjack]")
{
    games::blackjack game(7);
    for (int i = 0; i < 1000; i++) {
        auto s = game.reset();
        REQUIRE(s.player_sum >= 12);
        REQUIRE(s.player_sum <= 21);
        REQUIRE(s.dealer_showing >= 1);
        REQUIRE(s.dealer_showing <= 10);
        // hit until the game ends: busting is -1, otherwise the game continues
        for (;;) {
            auto out = game.step(games::blackjack::hit);
            if (out.terminated) {
                REQUIRE(out.reward == -1);
                break;
            }
            REQUIRE(out.reward == 0);
            REQUIRE(!out.next.natural);
        }
        // sticking always ends the game with -1, 0 or +1
        game.reset();
        auto out = game.step(games::blackjack::stick);
        REQUIRE(out.terminated);
        REQUIRE((out.reward == -1 || out.reward == 0 || out.reward == 1));
    }
}

TEST_CASE("blackjack: q_probabilistic learns the optimal strategy", "[games][blackjack]")
{
    using game_type = games::blackjack;
    using types = games::types<game_type>;
    game_type game(1);
    std::mt19937 prng(2);
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(1.0);
    train(game, learner, memory, 2000000, 0.1, 100, prng);

    solvers::blackjack exact;
    // the learnt decision matches the optimal one in almost every state; the few
    // exceptions are near-ties, where sampling noise in the estimates (a stick is
    // worth -1, 0 or +1) can put the two actions in the wrong order
    int disagreements = 0;
    for (int usable = 0; usable < 2; usable++) {
        for (unsigned int sum = 12; sum <= 21; sum++) {
            for (unsigned int dealer = 1; dealer <= 10; dealer++) {
                game_type::state_type s;
                s.player_sum = sum;
                s.usable_ace = usable == 1;
                s.dealer_showing = dealer;
                double q_stick = memory.value(types::state(s), types::action(game_type::stick));
                double q_hit   = memory.value(types::state(s), types::action(game_type::hit));
                double best_stick = exact.stick(sum, dealer);
                double best_hit   = exact.hit(sum, usable == 1, dealer);
                if ((q_stick >= q_hit) != (best_stick >= best_hit)) {
                    disagreements++;
                    INFO("sum " << sum << " usable " << usable << " dealer " << dealer);
                    REQUIRE(std::fabs(best_stick - best_hit) < 0.1);
                }
            }
        }
    }
    REQUIRE(disagreements <= 10);   // of 200 states

    // and it earns what the optimal strategy earns
    auto optimal = [&](const game_type::state_type & s, const std::vector<unsigned int> &) {
        double stick = s.natural ? exact.stick_natural(s.dealer_showing)
                                 : exact.stick(s.player_sum, s.dealer_showing);
        double hit = exact.hit(s.player_sum, s.usable_ace, s.dealer_showing);
        return stick >= hit ? game_type::stick : game_type::hit;
    };
    const unsigned int games_played = 200000;
    game_type eval_learnt(11), eval_optimal(11);
    double learnt = games::evaluate(eval_learnt, memory, games_played, 100, prng);
    double best = 0;
    for (unsigned int i = 0; i < games_played; i++) {
        best += games::play(eval_optimal, optimal, 100).total_reward;
    }
    best /= games_played;
    INFO("learnt " << learnt << " optimal " << best);
    REQUIRE(std::fabs(learnt - best) < 0.01);
}

// -------------------------------------------------------------- cliff walking

TEST_CASE("cliff walking: rules", "[games][cliff]")
{
    games::cliff_walking game;
    auto s = game.reset();
    REQUIRE(s == games::cliff_walking::start());
    // the border keeps the agent in place, at the usual cost
    auto out = game.step(games::left);
    REQUIRE(out.next == games::cliff_walking::start());
    REQUIRE(out.reward == -1);
    REQUIRE(!out.terminated);
    // the cliff costs -100 and sends the agent back to the start, without ending
    out = game.step(games::right);
    REQUIRE(out.reward == -100);
    REQUIRE(out.next == games::cliff_walking::start());
    REQUIRE(!out.terminated);
    // walking around the cliff reaches the goal in 13 moves
    game.reset();
    std::vector<unsigned int> path(1, games::up);
    path.insert(path.end(), 11, games::right);
    path.push_back(games::down);
    double total = 0;
    for (std::size_t i = 0; i < path.size(); i++) {
        out = game.step(path[i]);
        total += out.reward;
        REQUIRE(out.terminated == (i + 1 == path.size()));
    }
    REQUIRE(out.next == games::cliff_walking::goal());
    REQUIRE(total == -13);
}

TEST_CASE("cliff walking: q_learning learns the shortest path", "[games][cliff]")
{
    using game_type = games::cliff_walking;
    games::cliff_walking game;
    std::mt19937 prng(1);
    games::types<game_type>::policy memory;
    relearn::q_learning<games::types<game_type>::state, games::types<game_type>::action> learner;
    learner.alpha = 0.5;
    learner.gamma = 1.0;
    train(game, learner, memory, 1000, 0.1, 1000, prng);
    auto greedy = games::play(game,
                              [&](const game_type::state_type & s, const std::vector<unsigned int> & legal) {
                                  return games::greedy<game_type>(memory, s, legal, prng);
                              }, 100);
    REQUIRE(greedy.terminated);
    REQUIRE(greedy.steps == 13);
    REQUIRE(greedy.total_reward == -13);
}

// ------------------------------------------------------------ windy gridworld

namespace
{
/// exact model of the windy gridworld, written from the rules of Sutton & Barto
solvers::tabular windy_model(bool stochastic)
{
    using game_type = games::windy_gridworld;
    solvers::tabular m;
    m.states = game_type::width * game_type::height;
    m.actions = 4;
    m.gamma = 1;
    m.model = [stochastic](int s, int a) {
        std::vector<solvers::outcome> out;
        int x = s % game_type::width, y = s / game_type::width;
        if (x == 7 && y == 3) return out;                       // goal
        const int wind[] = {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};
        std::vector<int> pushes = {wind[x]};
        if (stochastic && wind[x] > 0) pushes = {wind[x] - 1, wind[x], wind[x] + 1};
        for (int push : pushes) {
            int nx = std::min(std::max(x + (a == 1) - (a == 3), 0), game_type::width - 1);
            int ny = std::min(std::max(y - (a == 0) + (a == 2), 0), game_type::height - 1);
            ny = std::min(std::max(ny - push, 0), game_type::height - 1);
            out.push_back({1.0 / pushes.size(), ny * game_type::width + nx, -1.0, nx == 7 && ny == 3});
        }
        return out;
    };
    return m;
}

/// the greedy action of @param memory in every state of a grid game
template <class game_type>
std::vector<int> greedy_policy(const typename games::types<game_type>::policy & memory,
                               int width, int height, std::mt19937 & prng)
{
    std::vector<int> policy(width * height);
    for (int s = 0; s < width * height; s++) {
        games::position p{s % width, s / width};
        policy[s] = games::greedy<game_type>(memory, p, {0, 1, 2, 3}, prng);
    }
    return policy;
}
} // namespace

TEST_CASE("windy gridworld: rules", "[games][windy]")
{
    using game_type = games::windy_gridworld;
    game_type game;
    REQUIRE(game.reset() == game_type::start());
    // the game agrees with the exact model on every move from every cell
    auto model = windy_model(false);
    for (int s = 0; s < model.states; s++) {
        games::position at{s % game_type::width, s / game_type::width};
        if (at == game_type::goal()) continue;
        for (unsigned int a = 0; a < 4; a++) {
            game.place(at);
            auto out = game.step(a);
            auto expected = model.model(s, static_cast<int>(a));
            REQUIRE(expected.size() == 1);
            REQUIRE(out.next.x + out.next.y * game_type::width == expected[0].next);
            REQUIRE(out.reward == expected[0].reward);
            REQUIRE(out.terminated == expected[0].terminal);
        }
    }
    // from the start, moving right is pushed up once it leaves column 3 (wind 1)
    game.reset();
    for (int i = 0; i < 3; i++) {
        REQUIRE(game.step(games::right).next == (games::position{i + 1, 3}));
    }
    REQUIRE(game.step(games::right).next == (games::position{4, 2}));
    // random wind only varies where there is wind: column 0 has none
    game_type random_wind(true, 3);
    for (int i = 0; i < 100; i++) {
        random_wind.place({0, 3});
        REQUIRE(random_wind.step(games::up).next == (games::position{0, 2}));
    }
}

TEST_CASE("windy gridworld: q_learning learns the shortest path", "[games][windy]")
{
    using game_type = games::windy_gridworld;
    game_type game;
    std::mt19937 prng(1);
    games::types<game_type>::policy memory;
    relearn::q_learning<games::types<game_type>::state, games::types<game_type>::action> learner;
    learner.alpha = 0.5;
    learner.gamma = 1.0;
    train(game, learner, memory, 500, 0.1, 1000, prng);
    auto exact = windy_model(false);
    REQUIRE(exact.optimal_values()[3 * game_type::width] == Approx(-15));
    auto greedy = games::play(game,
                              [&](const game_type::state_type & s, const std::vector<unsigned int> & legal) {
                                  return games::greedy<game_type>(memory, s, legal, prng);
                              }, 100);
    REQUIRE(greedy.terminated);
    REQUIRE(greedy.steps == 15);
}

TEST_CASE("windy gridworld: q_probabilistic learns the best policy in random wind", "[games][windy]")
{
    using game_type = games::windy_gridworld;
    game_type game(true, 1);
    std::mt19937 prng(11);
    games::types<game_type>::policy memory;
    relearn::q_probabilistic<games::types<game_type>::state, games::types<game_type>::action> learner(1.0);
    train(game, learner, memory, 2000, 0.1, 1000, prng);
    // the learnt greedy policy, evaluated exactly, needs as few moves as the optimal one
    auto exact = windy_model(true);
    auto best = exact.optimal_values();
    auto learnt = exact.policy_values(greedy_policy<game_type>(memory, game_type::width, game_type::height, prng));
    const int start = 3 * game_type::width;
    REQUIRE(learnt[start] == Approx(best[start]).margin(0.1));
}

// ---------------------------------------------------------------- frozen lake

namespace
{
/// exact model of FrozenLake, written from Gymnasium's rules
solvers::tabular frozen_lake_model(const std::vector<std::string> & map, double gamma, bool slippery)
{
    solvers::tabular m;
    const int rows = static_cast<int>(map.size()), cols = static_cast<int>(map[0].size());
    m.states = rows * cols;
    m.actions = 4;
    m.gamma = gamma;
    m.model = [map, rows, cols, slippery](int s, int a) {
        std::vector<solvers::outcome> out;
        char here = map[s / cols][s % cols];
        if (here == 'H' || here == 'G') return out;
        std::vector<int> slides = slippery ? std::vector<int>{-1, 0, 1} : std::vector<int>{0};
        for (int d : slides) {
            int b = (a + 4 + d) % 4, r = s / cols, c = s % cols;
            if (b == 0) c = std::max(c - 1, 0);          // left
            if (b == 1) r = std::min(r + 1, rows - 1);   // down
            if (b == 2) c = std::min(c + 1, cols - 1);   // right
            if (b == 3) r = std::max(r - 1, 0);          // up
            char t = map[r][c];
            out.push_back({1.0 / slides.size(), r * cols + c, t == 'G' ? 1.0 : 0.0, t == 'G' || t == 'H'});
        }
        return out;
    };
    return m;
}
} // namespace

TEST_CASE("frozen lake: rules", "[games][frozen_lake]")
{
    using game_type = games::frozen_lake;
    // without slipping, every move from every cell matches the exact model
    game_type ice(game_type::map_4x4(), false);
    auto model = frozen_lake_model(game_type::map_4x4(), 1.0, false);
    REQUIRE(ice.reset() == 0u);
    REQUIRE(ice.tile(15) == 'G');
    for (int s = 0; s < model.states; s++) {
        for (unsigned int a = 0; a < 4; a++) {
            auto expected = model.model(s, static_cast<int>(a));
            if (expected.empty()) continue;                 // hole or goal
            ice.place(static_cast<unsigned int>(s));
            auto out = ice.step(a);
            REQUIRE(static_cast<int>(out.next) == expected[0].next);
            REQUIRE(out.reward == expected[0].reward);
            REQUIRE(out.terminated == expected[0].terminal);
        }
    }
    // on slippery ice, "down" from cell 0 goes down, left (stays) or right, a third each
    game_type slippery(game_type::map_4x4(), true, 5);
    int down = 0, stay = 0, right = 0;
    const int n = 30000;
    for (int i = 0; i < n; i++) {
        slippery.place(0);
        auto next = slippery.step(game_type::down).next;
        if (next == 4) down++;
        else if (next == 0) stay++;
        else if (next == 1) right++;
    }
    REQUIRE(down + stay + right == n);
    REQUIRE(down  == Approx(n / 3.0).epsilon(0.05));
    REQUIRE(stay  == Approx(n / 3.0).epsilon(0.05));
    REQUIRE(right == Approx(n / 3.0).epsilon(0.05));
}

TEST_CASE("frozen lake: q_probabilistic learns the optimal policy", "[games][frozen_lake]")
{
    using game_type = games::frozen_lake;
    const double gamma = 0.95;
    game_type game(game_type::map_4x4(), true, 1);
    std::mt19937 prng(11);
    games::types<game_type>::policy memory;
    relearn::q_probabilistic<games::types<game_type>::state, games::types<game_type>::action> learner(gamma);
    train(game, learner, memory, 10000, 0.1, game.step_limit(), prng);
    // the learnt greedy policy, evaluated exactly, is worth as much as the optimal one
    auto exact = frozen_lake_model(game_type::map_4x4(), gamma, true);
    std::vector<int> policy(exact.states);
    for (int s = 0; s < exact.states; s++) {
        policy[s] = static_cast<int>(games::greedy<game_type>(memory, static_cast<unsigned int>(s), {0, 1, 2, 3}, prng));
    }
    double best = exact.optimal_values()[0];
    double learnt = exact.policy_values(policy)[0];
    REQUIRE(learnt == Approx(best).margin(0.005));
}
