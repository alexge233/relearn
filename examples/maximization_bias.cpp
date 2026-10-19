/**
 * The maximization bias example (Sutton & Barto, Example 6.7).
 *
 * Going left from A is worse on average, so the best ε-greedy agent (ε = 0.1)
 * goes left only 5% of the time. Q-learning takes the maximum over B's noisy
 * estimates, which is biased upwards, and goes left far more often for many
 * episodes. Double Q-learning, still to be added to relearn, fixes this.
 *
 * @see games/maximization_bias.hpp for the rules
 */
#include "games/maximization_bias.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

// set shortcuts to the game and to its relearn types
using game_type = games::maximization_bias;
using types     = games::types<game_type>;

int main()
{
    const unsigned int runs = 2000;
    const unsigned int episodes = 300;
    // how many runs went left in each episode
    std::vector<double> left(episodes, 0);

    // average over many runs, each a new agent learning from scratch
    for (unsigned int run = 0; run < runs; run++) {
        game_type game(10, run);
        std::mt19937 prng(run + 5000);
        types::policy memory;
        relearn::q_learning<types::state, types::action> learner;
        learner.alpha = 0.1;
        learner.gamma = 1.0;

        // the ε-greedy agent, with ε = 0.1
        auto agent = [&](const unsigned int & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        for (unsigned int i = 0; i < episodes; i++) {
            auto r = games::play(game, agent, 10);
            // going left takes two steps: A to B, then B to the end
            left[i] += r.steps == 2;
            learner(r.episode, memory, r.terminated);
        }
    }

    // print the share of runs going left, averaged over every 25 episodes
    std::cout << "share of episodes going left from A (the best possible is 5%):" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    for (unsigned int from = 0; from < episodes; from += 25) {
        double share = 0;
        for (unsigned int i = from; i < from + 25; i++) {
            share += left[i];
        }
        std::cout << "    episodes " << std::setw(3) << from + 1 << "-" << std::setw(3) << from + 25
                  << ": " << std::setw(5) << 100 * share / (25 * runs) << "%" << std::endl;
    }
    return EXIT_SUCCESS;
}
