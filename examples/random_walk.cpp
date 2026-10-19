/**
 * The random walk (Sutton & Barto, Examples 6.2 and 7.1): a prediction task.
 *
 * There is only one action, so nothing is decided: the agent learns how much
 * each state is worth. The true values are known exactly, so the example prints
 * the learnt ones next to them.
 *
 * @see games/random_walk.hpp for the rules
 */
#include "games/random_walk.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

// set shortcuts to the game and to its relearn types
using game_type = games::random_walk;
using types     = games::types<game_type>;

/**
 * @brief the agent of a prediction task: there is only one action, so it always takes it
 * @return action 0
 */
unsigned int only_action(const unsigned int &,
                         const std::vector<unsigned int> &)
{
    return 0u;
}

/**
 * @brief learn the values of @param walk from @param episodes walks, and print them
 * @return the largest error between a learnt value and the true one
 */
double predict(game_type & walk,
               unsigned int episodes)
{
    // the values we learn, and `q_probabilistic` with no discount (γ = 1)
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(1.0);
    for (unsigned int i = 0; i < episodes; i++) {
        auto r = games::play(walk, only_action, 100000);
        learner(r.episode, memory, r.terminated);
    }
    // print each learnt value next to the true one, and find the largest error
    double worst = 0;
    std::cout << walk.states() << " states, " << episodes << " episodes" << std::endl;
    std::cout << "    state   learnt   true" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (unsigned int s = 1; s <= walk.states(); s++) {
        double v = memory.value(types::state(s), types::action(0));
        worst = std::max(worst, std::fabs(v - walk.true_value(s)));
        std::cout << "    " << std::setw(5) << s
                  << "  " << std::setw(7) << v
                  << "  " << std::setw(6) << walk.true_value(s) << std::endl;
    }
    return worst;
}

int main()
{
    // Example 6.2: 5 states, leaving on the left is worth 0
    game_type short_walk(5, 0, 1, 1);
    // Example 7.1: 19 states, leaving on the left is worth -1
    game_type long_walk(19, -1, 1, 2);
    double e1 = predict(short_walk, 10000);
    std::cout << std::endl;
    double e2 = predict(long_walk, 10000);
    // the learnt values must be close to the true ones
    return (e1 < 0.05 && e2 < 0.1) ? EXIT_SUCCESS : EXIT_FAILURE;
}
