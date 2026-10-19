/**
 * Blackjack (Sutton & Barto, Example 5.1) learnt with `q_probabilistic`.
 *
 * Blackjack is stochastic: the same decision can win or lose depending on the
 * cards that follow. `q_probabilistic` estimates the expected reward and the
 * probability of each next state from experience, so it needs no learning rate.
 *
 * The agent explores with ε-greedy, then plays greedily. The learnt strategy is
 * printed in the layout of Sutton & Barto, Figure 5.2, where the optimal policy
 * can be compared.
 *
 * @see games/blackjack.hpp for the rules
 */
#include "games/blackjack.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

// set shortcuts to the game and to its relearn types
using game_type = games::blackjack;
using types     = games::types<game_type>;

/**
 * @brief play @param episodes greedy games of @param game
 * @param memory is the learnt policy
 * @param prng breaks ties between actions of equal value
 * @return {wins, draws, losses} as fractions of the games played
 */
std::vector<double> results(game_type & game,
                            const types::policy & memory,
                            unsigned int episodes,
                            std::mt19937 & prng)
{
    // the greedy agent: always takes the best known action
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::greedy<game_type>(memory, s, legal, prng);
    };
    // count wins, draws and losses
    std::vector<double> count(3, 0);
    for (unsigned int i = 0; i < episodes; i++) {
        auto r = games::play(game, agent, 100);
        if (r.total_reward > 0) {
            count[0] += 1;
        }
        else if (r.total_reward == 0) {
            count[1] += 1;
        }
        else {
            count[2] += 1;
        }
    }
    // turn the counts into fractions
    for (auto & c : count) {
        c /= episodes;
    }
    return count;
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

/**
 * @brief print the fractions of wins, draws and losses in @param r
 */
void print_results(const std::vector<double> & r)
{
    std::cout << std::fixed << std::setprecision(3)
              << "win " << r[0] << "  draw " << r[1] << "  loss " << r[2];
}

int main()
{
    const unsigned int episodes = 500000;
    const double epsilon = 0.1;
    // one game to learn from, and another one to test the learnt policy on
    game_type game(1);
    game_type test_game(2);
    std::mt19937 prng(3);
    // the policy we learn, and the probabilistic Q-learning algorithm that learns it (γ = 1)
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(1.0);

    // how well the agent plays before learning anything
    auto before = results(test_game, memory, 100000, prng);
    std::cout << "before learning: ";
    print_results(before);
    std::cout << std::endl;

    // the ε-greedy agent: explores with probability ε, otherwise takes the best known action
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, epsilon, prng);
    };

    // play and learn after every game, testing the greedy policy every 100000 games
    for (unsigned int i = 1; i <= episodes; i++) {
        auto r = games::play(game, agent, 100);
        learner(r.episode, memory, r.terminated);
        if (i % 100000 == 0) {
            auto now = results(test_game, memory, 100000, prng);
            std::cout << "after " << std::setw(6) << i << " games: ";
            print_results(now);
            std::cout << "  (average reward " << std::showpos << std::setprecision(4)
                      << now[0] - now[2] << std::noshowpos << ")" << std::endl;
        }
    }

    // the learnt strategy, to compare with Figure 5.2
    print_policy(memory, true);
    print_policy(memory, false);
    return EXIT_SUCCESS;
}
