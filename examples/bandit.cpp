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
#include <cstdio>
#include <cstdlib>

using game_type = games::bandit;
using types     = games::types<game_type>;

int main()
{
    const unsigned int runs = 500;
    const unsigned int pulls = 1000;
    const unsigned int checkpoints[] = {10, 100, 500, 1000};
    std::printf("share of runs pulling the best arm, after this many pulls:\n");
    std::printf("            %6u %6u %6u %6u\n", checkpoints[0], checkpoints[1], checkpoints[2], checkpoints[3]);
    double final_share = 0;
    for (double epsilon : {0.0, 0.01, 0.1}) {
        std::vector<double> best(pulls + 1, 0);
        for (unsigned int run = 0; run < runs; run++) {
            game_type game(10, run);
            std::mt19937 prng(run + 1000);
            types::policy memory;
            relearn::q_probabilistic<types::state, types::action> learner(1.0);
            for (unsigned int i = 1; i <= pulls; i++) {
                auto r = games::play(game,
                                     [&](const unsigned int & s, const std::vector<unsigned int> & arms) {
                                         return games::epsilon_greedy<game_type>(memory, s, arms, epsilon, prng);
                                     }, 1);
                best[i] += r.episode[0].action.trait() == game.best_arm();
                learner(r.episode, memory, r.terminated);
            }
        }
        std::printf("  ε = %.2f: ", epsilon);
        for (unsigned int c : checkpoints) {
            std::printf(" %5.1f%%", 100 * best[c] / runs);
        }
        std::printf("\n");
        final_share = best[pulls] / runs;
    }
    // ε = 0.1 explores enough to find the best arm in most runs
    return final_share > 0.7 ? EXIT_SUCCESS : EXIT_FAILURE;
}
