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
#include <cstdlib>
#include <iomanip>
#include <iostream>

// set shortcuts to the game and to its relearn types
using game_type = games::gamblers_problem;
using types     = games::types<game_type>;

int main()
{
    // a random starting capital, so that every capital gets visited
    game_type game(0.4, 0, 100, 1);
    std::mt19937 prng(2);
    // the values we learn, and `q_probabilistic` with no discount (γ = 1)
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(1.0);

    // the ε-greedy agent, with ε = 0.1
    auto agent = [&](const unsigned int & s,
                     const std::vector<unsigned int> & stakes) {
        return games::epsilon_greedy<game_type>(memory, s, stakes, 0.1, prng);
    };
    for (unsigned int i = 0; i < 200000; i++) {
        auto r = games::play(game, agent, 1000);
        learner(r.episode, memory, r.terminated);
    }

    // print the learnt value and the greedy stake of a few capitals
    std::cout << "capital  learnt value  greedy stake" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (unsigned int capital : {10u, 25u, 40u, 50u, 60u, 75u, 90u}) {
        double value = memory.best_value(types::state(capital));
        unsigned int stake = games::greedy<game_type>(memory, capital, game.actions(capital), prng);
        std::cout << std::setw(7) << capital
                  << "  " << std::setw(12) << value
                  << "  " << std::setw(12) << stake << std::endl;
    }
    std::cout << "the book's values: V(25) = 0.16, V(50) = 0.40, V(75) = 0.64" << std::endl;
    // the value of 50 must be close to the book's 0.4
    double v50 = memory.best_value(types::state(50));
    return std::fabs(v50 - 0.4) < 0.05 ? EXIT_SUCCESS : EXIT_FAILURE;
}
