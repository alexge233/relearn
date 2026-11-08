/**
 * FrozenLake (Gymnasium's FrozenLake-v1 rules) learnt with `q_probabilistic`.
 *
 * The ice is slippery: a move goes the chosen way only a third of the time.
 * The agent cannot simply walk to the goal; it learns moves that keep it away
 * from the holes even when it slides. The printed policy shows them as arrows.
 *
 * Run it with `--watch` to see the agent slide about on each lake: its first three episodes, then one in every
 * tenth of them, the cells coloured by the values learnt so far (have a look at docs/watch.md).
 *
 * @see games/frozen_lake.hpp for the rules
 */
#include "games/frozen_lake.hpp"
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
// set shortcuts to the game, to the game as it is watched, and to its relearn types
using rules     = games::frozen_lake;
using game_type = games::watched<rules, screen>;
using types     = games::types<game_type>;
// whether, and how, to watch the agent play (from the command line)
games::watch_options options;

/**
 * @brief learn to cross the lake of @param map, and print the greedy policy
 * @param episodes is the number of episodes to learn from
 * @param prng draws the random choices of the agent
 * @return the success rate of the greedy policy
 */
double learn(const std::vector<std::string> & map,
             unsigned int episodes,
             std::mt19937 & prng)
{
    game_type game{rules(map, true, 1)};
    const rules & lake = game.inner();
    // the policy we learn, and the probabilistic Q-learning algorithm that learns it (γ = 0.95)
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(0.95);
    // watched, the cells are coloured by the values being learnt
    options.apply(game);
    game.memory = &memory;
    game.title  = std::to_string(lake.rows()) + "x" + std::to_string(lake.columns()) + " FrozenLake, q_probabilistic";
    // the ε-greedy agent: explores 10% of the time, otherwise takes the best known move
    auto agent = [&](const unsigned int & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
    };
    // play and learn after every episode; watched: the first three episodes, then one in every tenth of them
    for (unsigned int i = 0; i < episodes; i++) {
        game.on = options.on && (i < 3 || (i + 1) % (episodes / 10) == 0);
        auto r = games::play(game, agent, lake.step_limit());
        learner(r.episode, memory, r.terminated);
    }
    game.on = false;
    // the reward is 1 only when the goal is reached: the average return is the success rate
    double success = games::evaluate(game, memory, 10000, lake.step_limit(), prng);

    // print the success rate, and the greedy move of every cell as an arrow
    const char arrow[] = {'<', 'v', '>', '^'};
    std::cout << lake.rows() << "x" << lake.columns() << " lake after " << episodes
              << " episodes: the greedy policy reaches the goal in "
              << std::fixed << std::setprecision(1) << 100 * success << "% of games" << std::endl;
    for (int r = 0; r < lake.rows(); r++) {
        std::cout << "    ";
        for (int c = 0; c < lake.columns(); c++) {
            unsigned int s = static_cast<unsigned int>(r * lake.columns() + c);
            char t = lake.tile(s);
            // holes and the goal are shown as they are: there is no move to make there
            if (t == 'H' || t == 'G') {
                std::cout << t << " ";
            }
            else {
                auto move = games::greedy<game_type>(memory, s, game.actions(s), prng);
                std::cout << arrow[move] << " ";
            }
        }
        std::cout << std::endl;
    }
    return success;
}

int main(int argc,
         char ** argv)
{
    options = games::watch_options::parse(argc, argv);
    std::mt19937 prng(2);
    // the small lake, then the large one
    double small = learn(rules::map_4x4(), 10000, prng);
    std::cout << std::endl;
    double large = learn(rules::map_8x8(), 50000, prng);
    return (small > 0.5 && large > 0.5) ? EXIT_SUCCESS : EXIT_FAILURE;
}
