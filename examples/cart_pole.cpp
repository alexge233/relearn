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
#include <cstdio>
#include <cstdlib>

using game_type = games::discretised<games::cart_pole>;
using types     = games::types<game_type>;

int main()
{
    // position, velocity, angle (radians), angular velocity
    game_type game(games::cart_pole(1), {{-0.8, 0.8, 3}, {-0.5, 0.5, 3}, {-0.21, 0.21, 6}, {-0.87, 0.87, 3}});
    std::mt19937 prng(1);
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.1;
    learner.gamma = 0.99;

    std::printf("average steps balanced (the limit is %zu), per 1000 episodes:\n   ", game.inner().step_limit());
    double sum = 0, first = 0, last = 0;
    for (unsigned int i = 1; i <= 10000; i++) {
        auto r = games::play(game,
                             [&](const unsigned int & s, const std::vector<unsigned int> & legal) {
                                 return games::epsilon_greedy<game_type>(memory, s, legal, 0.05, prng);
                             }, game.inner().step_limit());
        learner(r.episode, memory, r.terminated);
        sum += r.steps;
        if (i % 1000 == 0) {
            std::printf(" %.0f", sum / 1000);
            if (i == 1000) first = sum / 1000;
            last = sum / 1000;
            sum = 0;
        }
    }
    std::printf("\n");
    return last > 2 * first ? EXIT_SUCCESS : EXIT_FAILURE;
}
