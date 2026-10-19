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
#include <cstdio>
#include <cstdlib>

using game_type = games::maximization_bias;
using types     = games::types<game_type>;

int main()
{
    const unsigned int runs = 2000;
    const unsigned int episodes = 300;
    std::vector<double> left(episodes, 0);
    for (unsigned int run = 0; run < runs; run++) {
        game_type game(10, run);
        std::mt19937 prng(run + 5000);
        types::policy memory;
        relearn::q_learning<types::state, types::action> learner;
        learner.alpha = 0.1;
        learner.gamma = 1.0;
        for (unsigned int i = 0; i < episodes; i++) {
            auto r = games::play(game,
                                 [&](const unsigned int & s, const std::vector<unsigned int> & legal) {
                                     return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
                                 }, 10);
            left[i] += r.steps == 2;     // going left takes two steps
            learner(r.episode, memory, r.terminated);
        }
    }
    std::printf("share of episodes going left from A (the best possible is 5%%):\n");
    for (unsigned int from = 0; from < episodes; from += 25) {
        double share = 0;
        for (unsigned int i = from; i < from + 25; i++) share += left[i];
        std::printf("    episodes %3u-%3u: %5.1f%%\n", from + 1, from + 25, 100 * share / (25 * runs));
    }
    return EXIT_SUCCESS;
}
