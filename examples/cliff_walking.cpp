/**
 * Cliff Walking (Sutton & Barto, Example 6.6) learnt with `q_learning`.
 *
 * Q-learning learns the shortest path, right along the edge of the cliff
 * (13 moves). While it explores with ε-greedy it still falls off now and then,
 * so its return during training stays below that of the safer path SARSA would
 * learn: the classic on-policy vs off-policy example.
 *
 * @see games/cliff_walking.hpp for the rules
 */
#include "games/cliff_walking.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

// set shortcuts to the game and to its relearn types
using game_type = games::cliff_walking;
using types     = games::types<game_type>;

/**
 * @brief draw the grid, marking with `*` the cells the greedy policy visits
 * @param memory is the learnt policy
 * @param prng breaks ties between actions of equal value
 */
void draw(const types::policy & memory,
          std::mt19937 & prng)
{
    // an empty grid, with the cliff along the bottom row
    std::vector<std::string> rows(game_type::height, std::string(game_type::width, '.'));
    for (int x = 1; x < game_type::width - 1; x++) {
        rows[game_type::height - 1][x] = 'C';
    }
    // follow the greedy policy from the start, marking each cell visited
    game_type game;
    auto s = game.reset();
    for (int step = 0; step < 100; step++) {
        rows[s.y][s.x] = '*';
        auto move = games::greedy<game_type>(memory, s, game.actions(s), prng);
        auto out  = game.step(move);
        s = out.next;
        if (out.terminated) {
            break;
        }
    }
    // mark the start and the goal, then print the grid
    rows[game_type::start().y][game_type::start().x] = 'S';
    rows[game_type::goal().y][game_type::goal().x]   = 'G';
    for (const auto & row : rows) {
        std::cout << "    " << row << std::endl;
    }
}

int main()
{
    const unsigned int episodes = 3000;
    const double epsilon = 0.1;
    game_type game;
    std::mt19937 prng(1);
    // the policy we learn, and the Q-learning algorithm that learns it
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.5;
    learner.gamma = 1.0;

    // the ε-greedy agent: explores with probability ε, otherwise takes the best known move
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, epsilon, prng);
    };

    // play and learn, printing the average return every 300 episodes
    double sum = 0;
    std::cout << std::fixed << std::setprecision(1);
    for (unsigned int i = 1; i <= episodes; i++) {
        auto r = games::play(game, agent, 1000);
        learner(r.episode, memory, r.terminated);
        sum += r.total_reward;
        if (i % 300 == 0) {
            std::cout << "episodes " << std::setw(4) << i - 299 << "-" << std::setw(4) << i
                      << ": average return while exploring " << std::setw(7) << sum / 300 << std::endl;
            sum = 0;
        }
    }

    // now play greedily: the learnt path should be the 13 moves along the cliff edge
    double greedy = games::evaluate(game, memory, 1, 100, prng);
    std::cout << std::endl << std::setprecision(0)
              << "greedy policy: return " << greedy
              << " (the shortest path is 13 moves, -13)" << std::endl;
    draw(memory, prng);
    return greedy == -13 ? EXIT_SUCCESS : EXIT_FAILURE;
}
