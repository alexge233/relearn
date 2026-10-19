/**
 * The 10-armed bandit testbed (Sutton & Barto, Section 2.3).
 *
 * Each pull is a one-step episode, so `q_probabilistic` keeps the exact average
 * reward of every arm: the sample-average method of the book. The example
 * compares how often greedy (ε = 0) and ε-greedy agents pick the best arm,
 * averaged over 500 different bandits, as in Figure 2.2.
 *
 * @see games/bandit.hpp for the rules
 */
#include "games/bandit.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

// set shortcuts to the game and to its relearn types
using game_type = games::bandit;
using types     = games::types<game_type>;

int main()
{
    const unsigned int runs = 500;
    const unsigned int pulls = 1000;
    const unsigned int checkpoints[] = {10, 100, 500, 1000};
    std::cout << "share of runs pulling the best arm, after this many pulls:" << std::endl;
    std::cout << "           ";
    for (unsigned int c : checkpoints) {
        std::cout << " " << std::setw(6) << c;
    }
    std::cout << std::endl;

    double final_share = 0;
    for (double epsilon : {0.0, 0.01, 0.1}) {
        // best[i] counts the runs that pulled the best arm on pull i
        std::vector<double> best(pulls + 1, 0);
        for (unsigned int run = 0; run < runs; run++) {
            // a different bandit in every run
            game_type game(10, run);
            std::mt19937 prng(run + 1000);
            // the learnt average reward of each arm
            types::policy memory;
            relearn::q_probabilistic<types::state, types::action> learner(1.0);
            // the ε-greedy agent: explores with probability ε, otherwise pulls the best known arm
            auto agent = [&](const unsigned int & s,
                             const std::vector<unsigned int> & arms) {
                return games::epsilon_greedy<game_type>(memory, s, arms, epsilon, prng);
            };
            for (unsigned int i = 1; i <= pulls; i++) {
                // one pull is one episode
                auto r = games::play(game, agent, 1);
                if (r.episode[0].action.trait() == game.best_arm()) {
                    best[i] += 1;
                }
                learner(r.episode, memory, r.terminated);
            }
        }
        // print the share of runs that pulled the best arm at each checkpoint
        std::cout << "  ε = " << std::fixed << std::setprecision(2) << epsilon << ": ";
        std::cout << std::setprecision(1);
        for (unsigned int c : checkpoints) {
            std::cout << " " << std::setw(5) << 100 * best[c] / runs << "%";
        }
        std::cout << std::endl;
        final_share = best[pulls] / runs;
    }
    // ε = 0.1 explores enough to find the best arm in most runs
    return final_share > 0.7 ? EXIT_SUCCESS : EXIT_FAILURE;
}
