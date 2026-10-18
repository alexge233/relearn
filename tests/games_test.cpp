/**
 * Tests of the example games: their rules, and that relearn's learners reach
 * the known answer for each one (exact solutions from `solvers.hpp`, or the
 * optimal path lengths given by Sutton & Barto).
 * Random number generators are seeded, so results are reproducible.
 */
#include "../examples/games/blackjack.hpp"
#include "../examples/games/cliff_walking.hpp"
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
