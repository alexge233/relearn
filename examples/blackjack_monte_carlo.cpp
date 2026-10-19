/**
 * Blackjack (Sutton & Barto, Example 5.1) learnt with Monte Carlo control, in three ways:
 *
 * 1. Monte Carlo with exploring starts (Example 5.3, Figure 5.2): every game starts from a
 *    random situation with a random first move, then the agent plays greedily.
 * 2. On-policy Monte Carlo (Section 5.4): ordinary deals, and the agent explores ε-greedy.
 * 3. Off-policy Monte Carlo (Section 5.7): the agent plays completely at random, whilst the
 *    learner learns the greedy policy from those games, with weighted importance sampling.
 *
 * Each learnt greedy policy then plays 200000 ordinary games. The strategy learnt with
 * exploring starts is printed in the layout of Sutton & Barto, Figure 5.2.
 *
 * The program succeeds if all three earn more than -0.055 per game
 * (the optimal strategy earns about -0.043, playing at random about -0.4).
 *
 * @see games/blackjack.hpp for the rules, and blackjack.cpp for the same game learnt with `q_probabilistic`
 */
#include "games/blackjack.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

// set shortcuts to the game and to its relearn types
using game_type = games::blackjack;
using types     = games::types<game_type>;

/**
 * @brief learn with Monte Carlo and exploring starts
 * @param games_played is the number of games to learn from
 * @param prng chooses the random first moves, and breaks ties
 * @return the learnt values
 */
types::policy exploring_starts(unsigned int games_played,
                               std::mt19937 & prng)
{
    game_type game(1);
    game.exploring_starts(true);
    types::policy memory;
    relearn::monte_carlo<types::state, types::action> learner(1.0);
    // the first move of each game is random, the others greedy
    bool first = true;
    std::uniform_int_distribution<unsigned int> coin(0, 1);
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        if (first) {
            first = false;
            return coin(prng);
        }
        return games::greedy<game_type>(memory, s, legal, prng);
    };
    // play a game, and learn from it once it has ended
    for (unsigned int i = 0; i < games_played; i++) {
        first = true;
        auto r = games::play(game, agent, 100);
        learner(r.episode, memory, r.terminated);
    }
    return memory;
}

/**
 * @brief learn with on-policy Monte Carlo, exploring ε-greedy
 * @param games_played is the number of games to learn from
 * @param prng chooses the exploring moves, and breaks ties
 * @return the learnt values
 */
types::policy on_policy(unsigned int games_played,
                        std::mt19937 & prng)
{
    game_type game(2);
    types::policy memory;
    relearn::monte_carlo<types::state, types::action> learner(1.0);
    // the ε-greedy agent, with ε = 0.1
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
    };
    for (unsigned int i = 0; i < games_played; i++) {
        auto r = games::play(game, agent, 100);
        learner(r.episode, memory, r.terminated);
    }
    return memory;
}

/**
 * @brief learn the greedy policy with off-policy Monte Carlo, from games played at random
 * @param games_played is the number of games to learn from
 * @param prng chooses the random moves
 * @return the learnt values
 */
types::policy off_policy(unsigned int games_played,
                         std::mt19937 & prng)
{
    game_type game(3);
    types::policy memory;
    // the behaviour policy picks stick or hit with probability 1/2 each: b(a | s) = 0.5
    relearn::off_policy_monte_carlo<types::state, types::action> learner(1.0,
        [](const types::state &, const types::action &) {
            return 0.5;
        });
    std::uniform_int_distribution<unsigned int> coin(0, 1);
    auto random_player = [&](const game_type::state_type &,
                             const std::vector<unsigned int> &) {
        return coin(prng);
    };
    for (unsigned int i = 0; i < games_played; i++) {
        auto r = games::play(game, random_player, 100);
        learner(r.episode, memory, r.terminated);
    }
    return memory;
}

/**
 * @brief print the greedy action of every state, as in Sutton & Barto Figure 5.2
 * @param memory is the learnt policy
 * @param usable_ace selects the states with or without a usable Ace
 */
void print_policy(const types::policy & memory,
                  bool usable_ace)
{
    std::cout << std::endl << (usable_ace ? "With" : "No") << " usable ace  (S = stick, H = hit)" << std::endl;
    std::cout << "player \\ dealer:  A  2  3  4  5  6  7  8  9 10" << std::endl;
    // one row per player total, from 21 down to 12
    for (unsigned int sum = 21; sum >= 12; sum--) {
        std::cout << std::setw(15) << sum << ": ";
        // one column per dealer card, from Ace to 10
        for (unsigned int dealer = 1; dealer <= 10; dealer++) {
            game_type::state_type s;
            s.player_sum     = sum;
            s.usable_ace     = usable_ace;
            s.dealer_showing = dealer;
            // the greedy action is the one of higher value (stick on a tie)
            double q_stick = memory.value(types::state(s), types::action(game_type::stick));
            double q_hit   = memory.value(types::state(s), types::action(game_type::hit));
            std::cout << " " << (q_stick >= q_hit ? 'S' : 'H') << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    const unsigned int games_played = 500000;
    std::mt19937 prng(4);
    std::cout << "Blackjack, learnt with Monte Carlo control from " << games_played << " games each" << std::endl;

    // learn in each of the three ways
    auto es  = exploring_starts(games_played, prng);
    auto on  = on_policy(games_played, prng);
    auto off = off_policy(games_played, prng);

    // then each greedy policy plays the same 200000 ordinary games
    std::cout << std::endl << "average reward per game of the greedy policy learnt:" << std::endl;
    std::cout << std::fixed << std::setprecision(4) << std::showpos;
    const std::vector<std::pair<std::string, const types::policy *>> learnt = {
        {"exploring starts       ", &es},
        {"on-policy, ε = 0.1     ", &on},
        {"off-policy, from random", &off}
    };
    bool good = true;
    for (const auto & method : learnt) {
        game_type test_game(5);
        double reward = games::evaluate(test_game, *method.second, 200000, 100, prng);
        std::cout << "    " << method.first << "  " << reward << std::endl;
        good = good && reward > -0.055;
    }
    std::cout << std::noshowpos;

    // the strategy learnt with exploring starts, to compare with Figure 5.2
    print_policy(es, true);
    print_policy(es, false);
    return good ? EXIT_SUCCESS : EXIT_FAILURE;
}
