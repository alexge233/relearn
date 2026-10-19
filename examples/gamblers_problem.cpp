/**
 * The gambler's problem (Sutton & Barto, Example 4.3) learnt with `q_probabilistic`.
 *
 * The value of a capital is the probability of reaching 100 from it. The
 * example prints the learnt values next to the ones the book derives:
 * with a coin that lands heads 40% of the time, V(25) = 0.16, V(50) = 0.4
 * and V(75) = 0.64. Many stakes are equally good, so the learnt stakes may
 * differ from the book's figure while being just as good.
 *
 * @see games/gamblers_problem.hpp for the rules
 */
#include "games/gamblers_problem.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

using game_type = games::gamblers_problem;
using types     = games::types<game_type>;

int main()
{
    game_type game(0.4, 0, 100, 1);   // random starting capital: every capital gets visited
    std::mt19937 prng(2);
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(1.0);
    for (unsigned int i = 0; i < 200000; i++) {
        auto r = games::play(game,
                             [&](const unsigned int & s, const std::vector<unsigned int> & stakes) {
                                 return games::epsilon_greedy<game_type>(memory, s, stakes, 0.1, prng);
                             }, 1000);
        learner(r.episode, memory, r.terminated);
    }
    std::printf("capital  learnt value  greedy stake\n");
    for (unsigned int capital : {10u, 25u, 40u, 50u, 60u, 75u, 90u}) {
        std::printf("%7u  %12.3f  %12u\n", capital, memory.best_value(types::state(capital)),
                    games::greedy<game_type>(memory, capital, game.actions(capital), prng));
    }
    std::printf("the book's values: V(25) = 0.16, V(50) = 0.40, V(75) = 0.64\n");
    double v50 = memory.best_value(types::state(50));
    return std::fabs(v50 - 0.4) < 0.05 ? EXIT_SUCCESS : EXIT_FAILURE;
}
