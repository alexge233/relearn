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
#include <cstdio>
#include <cstdlib>

using game_type = games::blackjack;
using types     = games::types<game_type>;

/// play @param episodes greedy games: @return {wins, draws, losses} as fractions
std::vector<double> results(game_type & game,
                            const types::policy & memory,
                            unsigned int episodes,
                            std::mt19937 & prng)
{
    std::vector<double> count(3, 0);
    for (unsigned int i = 0; i < episodes; i++) {
        auto r = games::play(game,
                             [&](const game_type::state_type & s, const std::vector<unsigned int> & legal) {
                                 return games::greedy<game_type>(memory, s, legal, prng);
                             }, 100);
        count[r.total_reward > 0 ? 0 : (r.total_reward == 0 ? 1 : 2)] += 1;
    }
    for (auto & c : count) {
        c /= episodes;
    }
    return count;
}

/// print the greedy action of every state, as in Sutton & Barto Figure 5.2
void print_policy(const types::policy & memory, bool usable_ace)
{
    std::printf("\n%s usable ace  (S = stick, H = hit)\n", usable_ace ? "With" : "No");
    std::printf("player \\ dealer:  A  2  3  4  5  6  7  8  9 10\n");
    for (unsigned int sum = 21; sum >= 12; sum--) {
        std::printf("%15u: ", sum);
        for (unsigned int dealer = 1; dealer <= 10; dealer++) {
            game_type::state_type s;
            s.player_sum = sum;
            s.usable_ace = usable_ace;
            s.dealer_showing = dealer;
            double q_stick = memory.value(types::state(s), types::action(game_type::stick));
            double q_hit   = memory.value(types::state(s), types::action(game_type::hit));
            std::printf(" %c ", q_stick >= q_hit ? 'S' : 'H');
        }
        std::printf("\n");
    }
}

int main()
{
    const unsigned int episodes = 500000;
    const double epsilon = 0.1;
    game_type game(1);
    game_type test_game(2);
    std::mt19937 prng(3);
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(1.0);

    auto before = results(test_game, memory, 100000, prng);
    std::printf("before learning: win %.3f  draw %.3f  loss %.3f\n", before[0], before[1], before[2]);

    for (unsigned int i = 1; i <= episodes; i++) {
        auto r = games::play(game,
                             [&](const game_type::state_type & s, const std::vector<unsigned int> & legal) {
                                 return games::epsilon_greedy<game_type>(memory, s, legal, epsilon, prng);
                             }, 100);
        learner(r.episode, memory, r.terminated);
        if (i % 100000 == 0) {
            auto now = results(test_game, memory, 100000, prng);
            std::printf("after %6u games: win %.3f  draw %.3f  loss %.3f  (average reward %+.4f)\n",
                        i, now[0], now[1], now[2], now[0] - now[2]);
        }
    }
    print_policy(memory, true);
    print_policy(memory, false);
    return EXIT_SUCCESS;
}
