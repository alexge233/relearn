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
 * Run it with `--watch` to see the car: its first three episodes, then every 500th, next to the (position,
 * velocity) plane coloured by the values learnt so far, the book's Figure 10.1 (have a look at docs/watch.md).
 *
 * @see games/mountain_car.hpp for the physics
 */
#include "games/mountain_car.hpp"
#include "games/discretised.hpp"
#include "games/views.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

// the frames go to the terminal, or, built by the window sub-project, to a raylib window
#ifdef RELEARN_WINDOW
#include "window/window.hpp"
using screen = games::window_screen;
#else
using screen = games::terminal_screen;
#endif
// set shortcuts to the game, cut into a grid, to the game as it is watched, and to its relearn types
using rules     = games::discretised<games::mountain_car>;
using game_type = games::watched<rules, screen>;
using types     = games::types<game_type>;

int main(int argc,
         char ** argv)
{
    const auto options = games::watch_options::parse(argc, argv);
    // cut position and velocity into 20 bins each
    game_type game{rules(games::mountain_car(1), {{-1.2, 0.6, 20}, {-0.07, 0.07, 20}})};
    std::mt19937 prng(1);
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.1;
    learner.gamma = 1.0;
    // watched, the plane of positions and velocities is coloured by the values being learnt
    options.apply(game);
    game.memory = &memory;
    game.title  = "Mountain Car, q_learning";

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
        // watched: the first three episodes, then every 500th
        game.on = options.on && (i <= 3 || i % 500 == 0);
        auto r = games::play(game, agent, game.inner().inner().step_limit());
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
