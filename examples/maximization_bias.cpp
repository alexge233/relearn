/**
 * The maximization bias example (Sutton & Barto, Example 6.7 and Figure 6.5).
 *
 * Going left from A is worse on average, so the best ε-greedy agent (ε = 0.1)
 * goes left only 5% of the time. Q-learning takes the maximum over B's noisy
 * estimates, which is biased upwards, and goes left far more often for many
 * episodes. Double Q-learning picks B's best action with one estimate and values
 * it with another, which removes the bias: it soon goes left about as rarely as it should.
 *
 * @see games/maximization_bias.hpp for the rules
 * @see docs/double_q_learning.md for the algorithm
 */
#include "games/maximization_bias.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

// set shortcuts to the game and to its relearn types
using game_type = games::maximization_bias;
using types     = games::types<game_type>;

// the settings of Sutton & Barto, Example 6.7
const unsigned int runs     = 2000;
const unsigned int episodes = 300;

/**
 * @brief Q-learning has no coin: nothing to seed
 * @param learner is the learner
 * @param seed is not used
 */
template <class learner_type>
void seed_coin(learner_type &,
               unsigned int)
{}

/**
 * @brief seed Double Q-learning's coin, so that each run flips its own coins
 * @param learner is the learner
 * @param seed is the seed of the coin
 */
void seed_coin(relearn::double_q_learning<types::state, types::action> & learner,
               unsigned int seed)
{
    learner.coin.seed(seed);
}

/**
 * @brief train a learner in `runs` runs of `episodes` episodes each
 * @return for each episode, the share of runs that went left from A
 */
template <class learner_type>
std::vector<double> share_left()
{
    std::vector<double> left(episodes, 0);
    // average over many runs, each a new agent learning from scratch
    for (unsigned int run = 0; run < runs; run++) {
        game_type game(10, run);
        std::mt19937 prng(run + 5000);
        types::policy memory;
        learner_type learner;
        learner.alpha = 0.1;
        learner.gamma = 1.0;
        seed_coin(learner, run + 9000);

        // the ε-greedy agent, with ε = 0.1
        auto agent = [&](const unsigned int & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        for (unsigned int i = 0; i < episodes; i++) {
            auto r = games::play(game, agent, 10);
            // going left takes two steps: A to B, then B to the end
            left[i] += (r.steps == 2) / static_cast<double>(runs);
            learner(r.episode, memory, r.terminated);
        }
    }
    return left;
}

int main()
{
    auto q        = share_left<relearn::q_learning<types::state, types::action>>();
    auto double_q = share_left<relearn::double_q_learning<types::state, types::action>>();

    // print the share of runs going left, averaged over every 25 episodes
    std::cout << "share of episodes going left from A (the best possible is 5%), "
              << runs << " runs:" << std::endl;
    std::cout << "    episodes   Q-learning   Double Q-learning" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    double q_late = 0;
    double double_q_late = 0;
    for (unsigned int from = 0; from < episodes; from += 25) {
        double share_q = 0;
        double share_double_q = 0;
        for (unsigned int i = from; i < from + 25; i++) {
            share_q        += q[i] / 25;
            share_double_q += double_q[i] / 25;
        }
        std::cout << "     " << std::setw(3) << from + 1 << "-" << std::setw(3) << from + 25
                  << "   " << std::setw(9) << 100 * share_q << "%"
                  << "   " << std::setw(16) << 100 * share_double_q << "%" << std::endl;
        // keep the last 100 episodes, to check the result
        if (from >= episodes - 100) {
            q_late        += share_q / 4;
            double_q_late += share_double_q / 4;
        }
    }
    // as in the book: Q-learning goes left far too often, Double Q-learning about 5% of the time
    bool as_expected = (q_late > 0.15) && (double_q_late < 0.09);
    return as_expected ? EXIT_SUCCESS : EXIT_FAILURE;
}
