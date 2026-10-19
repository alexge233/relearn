/**
 * FrozenLake (Gymnasium's FrozenLake-v1 rules) learnt with `q_probabilistic`.
 *
 * The ice is slippery: a move goes the chosen way only a third of the time.
 * The agent cannot simply walk to the goal; it learns moves that keep it away
 * from the holes even when it slides. The printed policy shows them as arrows.
 *
 * @see games/frozen_lake.hpp for the rules
 */
#include "games/frozen_lake.hpp"
#include <cstdio>
#include <cstdlib>

using game_type = games::frozen_lake;
using types     = games::types<game_type>;

/// train on @param map, print the greedy policy: @return its success rate
double learn(const std::vector<std::string> & map, unsigned int episodes, std::mt19937 & prng)
{
    game_type game(map, true, 1);
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(0.95);
    for (unsigned int i = 0; i < episodes; i++) {
        auto r = games::play(game,
                             [&](const unsigned int & s, const std::vector<unsigned int> & legal) {
                                 return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
                             }, game.step_limit());
        learner(r.episode, memory, r.terminated);
    }
    // the reward is 1 only when the goal is reached: the average return is the success rate
    double success = games::evaluate(game, memory, 10000, game.step_limit(), prng);

    const char arrow[] = {'<', 'v', '>', '^'};
    std::printf("%dx%d lake after %u episodes: the greedy policy reaches the goal in %.1f%% of games\n",
                game.rows(), game.columns(), episodes, 100 * success);
    for (int r = 0; r < game.rows(); r++) {
        std::printf("    ");
        for (int c = 0; c < game.columns(); c++) {
            unsigned int s = static_cast<unsigned int>(r * game.columns() + c);
            char t = game.tile(s);
            if (t == 'H' || t == 'G') {
                std::printf("%c ", t);
            }
            else {
                std::printf("%c ", arrow[games::greedy<game_type>(memory, s, game.actions(s), prng)]);
            }
        }
        std::printf("\n");
    }
    return success;
}

int main()
{
    std::mt19937 prng(2);
    double small = learn(game_type::map_4x4(), 10000, prng);
    std::printf("\n");
    double large = learn(game_type::map_8x8(), 50000, prng);
    return (small > 0.5 && large > 0.5) ? EXIT_SUCCESS : EXIT_FAILURE;
}
