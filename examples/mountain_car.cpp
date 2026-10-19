/**
 * Mountain car (Gymnasium's MountainCar-v0 physics) learnt with `q_learning`.
 *
 * The state (position, velocity) is cut into a 20 x 20 grid. The agent acts
 * greedily without random moves: every value starts at 0 and every step costs
 * -1, so moves it has not tried look best, and it explores by itself
 * (optimistic initial values, Sutton & Barto Section 2.6).
 *
 * At first no episode reaches the flag within Gymnasium's 200 steps (return
 * -200); the agent learns to rock the car back and forth to build up speed.
 *
 * @see games/mountain_car.hpp for the physics
 */
#include "games/mountain_car.hpp"
#include "games/discretised.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

// set shortcuts to the game, cut into a grid, and to its relearn types
using game_type = games::discretised<games::mountain_car>;
using types     = games::types<game_type>;

int main()
{
    // cut position and velocity into 20 bins each
    game_type game(games::mountain_car(1), {{-1.2, 0.6, 20}, {-0.07, 0.07, 20}});
    std::mt19937 prng(1);
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.1;
    learner.gamma = 1.0;

    // the greedy agent: untried moves are worth 0, more than any tried one, so it explores
    auto agent = [&](const unsigned int & s,
                     const std::vector<unsigned int> & legal) {
        return games::greedy<game_type>(memory, s, legal, prng);
    };

    // play and learn, printing the average return every 500 episodes
    std::cout << "average return (-200: the flag is never reached), per 500 episodes:" << std::endl << "   ";
    std::cout << std::fixed << std::setprecision(0);
    double sum  = 0;
    double last = 0;
    for (unsigned int i = 1; i <= 5000; i++) {
        auto r = games::play(game, agent, game.inner().step_limit());
        learner(r.episode, memory, r.terminated);
        sum += r.total_reward;
        if (i % 500 == 0) {
            std::cout << " " << sum / 500;
            last = sum / 500;
            sum = 0;
        }
    }
    std::cout << std::endl;
    // at the end the agent must reach the flag in fewer than 180 steps on average
    return last > -180 ? EXIT_SUCCESS : EXIT_FAILURE;
}
