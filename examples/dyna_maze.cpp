/**
 * The Dyna maze (Sutton & Barto, Example 8.1) and the blocking maze (Example 8.2)
 * learnt with `q_learning`.
 *
 * 1. Steps per episode fall as the agent learns the way to the goal, as in
 *    Figure 8.2 (the curve without planning).
 * 2. In the blocking maze the short way closes after 10000 moves, and the
 *    cumulative reward stops growing: plain Q-learning does not find the new
 *    way. Nothing in the maze is ever worth less than 0, so the values along
 *    the blocked route only fade towards 0, and still beat every move the agent
 *    has not tried (also 0); 10% of random moves rarely make the 16-move detour.
 *    This is the case Dyna-Q+ (Figure 8.4), still to be added, is designed for.
 *
 * @see games/dyna_maze.hpp for the layouts
 */
#include "games/dyna_maze.hpp"
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>

// set shortcuts to the game and to its relearn types
using game_type = games::dyna_maze;
using types     = games::types<game_type>;

int main()
{
    std::mt19937 prng(1);

    // 1. steps per episode in the maze
    game_type maze;
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.5;
    learner.gamma = 0.95;
    std::cout << "maze (shortest path 14 moves):" << std::endl;
    for (const auto & row : maze.draw()) {
        std::cout << "    " << row << std::endl;
    }

    // the ε-greedy agent, with ε = 0.1
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
    };
    // play 50 episodes, printing the moves taken in the first 5 and then every 5th
    std::cout << "moves per episode:";
    for (unsigned int i = 1; i <= 50; i++) {
        auto r = games::play(maze, agent, 10000);
        learner(r.episode, memory, r.terminated);
        if (i <= 5 || i % 5 == 0) {
            std::cout << " " << r.steps;
        }
    }
    // then follow the learnt policy greedily
    auto greedy_agent = [&](const game_type::state_type & s,
                            const std::vector<unsigned int> & legal) {
        return games::greedy<game_type>(memory, s, legal, prng);
    };
    auto greedy = games::play(maze, greedy_agent, 100);
    std::cout << std::endl << "greedy policy after 50 episodes: " << greedy.steps << " moves" << std::endl;

    // 2. the blocking maze changes after 10000 moves
    const unsigned long change = 10000;
    const unsigned long total  = 20000;
    game_type blocking(game_type::blocking, change);
    types::policy memory_b;
    std::cout << std::endl << "blocking maze, cumulative reward every 1000 moves (the wall moves at "
              << change << "):" << std::endl << "   ";

    // the ε-greedy agent of the blocking maze, learning its own policy
    auto agent_b = [&](const game_type::state_type & s,
                       const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory_b, s, legal, 0.1, prng);
    };
    double reward = 0;
    unsigned long next_report = 1000;
    std::cout << std::fixed << std::setprecision(0);
    while (blocking.moves() < total) {
        // episodes are cut at 500 moves, and still learnt (see `terminated`)
        auto r = games::play(blocking, agent_b, std::min<unsigned long>(500, total - blocking.moves()));
        learner(r.episode, memory_b, r.terminated);
        reward += r.total_reward;
        // print the cumulative reward each time another 1000 moves have been made
        while (blocking.moves() >= next_report && next_report <= total) {
            std::cout << " " << reward;
            next_report += 1000;
        }
    }
    std::cout << std::endl;
    return greedy.terminated ? EXIT_SUCCESS : EXIT_FAILURE;
}
