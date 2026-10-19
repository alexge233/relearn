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
#include <cmath>
#include <cstdio>
#include <cstdlib>

using game_type = games::random_walk;
using types     = games::types<game_type>;

/// learn the values of @param walk: @return the largest error
double predict(game_type & walk, unsigned int episodes)
{
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(1.0);
    for (unsigned int i = 0; i < episodes; i++) {
        auto r = games::play(walk, [](const unsigned int &, const std::vector<unsigned int> &) { return 0u; }, 100000);
        learner(r.episode, memory, r.terminated);
    }
    double worst = 0;
    std::printf("%u states, %u episodes\n    state   learnt   true\n", walk.states(), episodes);
    for (unsigned int s = 1; s <= walk.states(); s++) {
        double v = memory.value(types::state(s), types::action(0));
        worst = std::max(worst, std::fabs(v - walk.true_value(s)));
        std::printf("    %5u  %7.3f  %6.3f\n", s, v, walk.true_value(s));
    }
    return worst;
}

int main()
{
    game_type short_walk(5, 0, 1, 1);       // Example 6.2
    game_type long_walk(19, -1, 1, 2);      // Example 7.1
    double e1 = predict(short_walk, 10000);
    std::printf("\n");
    double e2 = predict(long_walk, 10000);
    return (e1 < 0.05 && e2 < 0.1) ? EXIT_SUCCESS : EXIT_FAILURE;
}
