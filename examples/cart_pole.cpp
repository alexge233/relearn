/**
 * Cart-pole (Gymnasium's CartPole-v1 physics) learnt with `q_learning`.
 *
 * The state is continuous, so it is cut into 162 boxes, after Barto, Sutton &
 * Anderson (1983). Episodes that reach Gymnasium's limit of 500 steps are cut
 * short, not failed: they are learnt with `terminated = false`.
 *
 * Boxes lump different states together, which limits how well a table can do;
 * function approximation (still to be added) removes that limit.
 *
 * @see games/cart_pole.hpp for the physics
 */
#include "games/cart_pole.hpp"
#include "games/discretised.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

// set shortcuts to the game, cut into boxes, and to its relearn types
using game_type = games::discretised<games::cart_pole>;
using types     = games::types<game_type>;

int main()
{
    // cut each variable into bins: position, velocity, angle (radians), angular velocity
    game_type game(games::cart_pole(1), {{-0.8, 0.8, 3}, {-0.5, 0.5, 3}, {-0.21, 0.21, 6}, {-0.87, 0.87, 3}});
    std::mt19937 prng(1);
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.1;
    learner.gamma = 0.99;

    // the ε-greedy agent, with ε = 0.05
    auto agent = [&](const unsigned int & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, 0.05, prng);
    };

    // play and learn, printing the average number of steps balanced every 1000 episodes
    std::cout << "average steps balanced (the limit is " << game.inner().step_limit()
              << "), per 1000 episodes:" << std::endl << "   ";
    std::cout << std::fixed << std::setprecision(0);
    double sum   = 0;
    double first = 0;
    double last  = 0;
    for (unsigned int i = 1; i <= 10000; i++) {
        auto r = games::play(game, agent, game.inner().step_limit());
        learner(r.episode, memory, r.terminated);
        sum += r.steps;
        if (i % 1000 == 0) {
            std::cout << " " << sum / 1000;
            if (i == 1000) {
                first = sum / 1000;
            }
            last = sum / 1000;
            sum = 0;
        }
    }
    std::cout << std::endl;
    // the agent must balance at least twice as long at the end as at the start
    return last > 2 * first ? EXIT_SUCCESS : EXIT_FAILURE;
}
