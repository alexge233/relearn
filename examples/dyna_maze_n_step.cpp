/**
 * The Dyna maze (Sutton & Barto, Example 8.1) learnt with n-step SARSA (Section 7.2).
 *
 * 1. Figure 7.4: after the first episode, one-step SARSA has learnt only the last move,
 *    the one that reached the goal, whereas 10-step SARSA has learnt the last 10 moves.
 * 2. The moves per episode, with n = 1, 2, 4, 8 and 16, averaged over 30 runs: the larger n,
 *    the faster the agent learns its way to the goal in the early episodes.
 * 3. The greedy path learnt after 50 episodes, next to the shortest one (14 moves).
 *
 * The program succeeds if 4-step and 8-step SARSA learn faster than one-step SARSA in
 * episodes 2 to 10, and if every greedy path reaches the goal in at most 30 moves.
 *
 * @see games/dyna_maze.hpp for the layout
 */
#include "games/dyna_maze.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// set shortcuts to the game, to its relearn types, and to the learner
using game_type    = games::dyna_maze;
using types        = games::types<game_type>;
using learner_type = relearn::n_step_sarsa<types::state, types::action>;

/**
 * @brief the maze, with an arrow on every cell where a move has a value above 0
 * @param maze is the maze
 * @param memory is the learnt policy
 * @return one string per row
 */
std::vector<std::string> draw_learnt(const game_type & maze,
                                     const types::policy & memory)
{
    auto rows = maze.draw();
    const char arrows[] = {'^', '>', 'v', '<'};
    for (int y = 0; y < game_type::height; y++) {
        for (int x = 0; x < game_type::width; x++) {
            for (unsigned int a = 0; a < 4; a++) {
                if (memory.value(types::state(games::position{x, y}), types::action(a)) > 0) {
                    rows[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] = arrows[a];
                }
            }
        }
    }
    return rows;
}

/**
 * @brief Figure 7.4: the moves learnt from the first episode, by one-step and by 10-step SARSA
 *
 * Every value is 0 until the goal is reached, so during the first episode nothing that is learnt
 * changes what the agent does: with the same seed, both agents walk the very same path.
 */
void figure_7_4()
{
    std::vector<std::vector<std::string>> drawn;
    std::size_t moves = 0;
    for (unsigned int n : {1u, 10u}) {
        // a seed whose first episode ends with 10 moves in 10 different cells, as in the figure
        std::mt19937 prng(32);
        game_type maze;
        types::policy memory;
        learner_type learner(0.1, 0.95, n);
        // the ε-greedy agent, with ε = 0.1 (ties between equal values are broken at random)
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        moves = games::play_online(maze, agent, learner, memory, 100000, n).steps;
        drawn.push_back(draw_learnt(maze, memory));
    }
    std::cout << "Figure 7.4: the moves learnt from the first episode (" << moves << " moves)" << std::endl;
    std::cout << "    one-step     10-step" << std::endl;
    for (std::size_t y = 0; y < drawn[0].size(); y++) {
        std::cout << "    " << drawn[0][y] << "    " << drawn[1][y] << std::endl;
    }
}

/**
 * @brief the moves of each episode, averaged over many runs
 * @param n is the number of steps of n-step SARSA
 * @param episodes is the number of episodes of each run
 * @param runs is the number of runs, each a new agent learning from scratch
 * @param greedy_moves receives, for each run, the moves of the greedy policy learnt
 * @return the mean moves of each episode
 *
 * The settings of Figure 8.2: ε-greedy with ε = 0.1, α = 0.1, γ = 0.95, learning after every move.
 */
std::vector<double> moves_per_episode(unsigned int n,
                                      unsigned int episodes,
                                      unsigned int runs,
                                      std::vector<std::size_t> & greedy_moves)
{
    std::vector<double> moves(episodes, 0);
    for (unsigned int run = 1; run <= runs; run++) {
        std::mt19937 prng(run);
        game_type maze;
        types::policy memory;
        learner_type learner(0.1, 0.95, n);
        // the ε-greedy agent, with ε = 0.1
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        // play the episodes, learning after every move: the learner is given the last n moves
        for (unsigned int i = 0; i < episodes; i++) {
            auto r = games::play_online(maze, agent, learner, memory, 100000, n);
            moves[i] += static_cast<double>(r.steps) / runs;
        }
        // then follow the learnt policy greedily, without exploring
        auto greedy_agent = [&](const game_type::state_type & s,
                                const std::vector<unsigned int> & legal) {
            return games::greedy<game_type>(memory, s, legal, prng);
        };
        greedy_moves.push_back(games::play(maze, greedy_agent, 100).steps);
    }
    return moves;
}

int main()
{
    // 1. Figure 7.4: what the first episode teaches
    figure_7_4();

    // 2. moves per episode, for each n, over 30 runs
    std::cout << std::endl << "Moves per episode (shortest path 14 moves), 30 runs:" << std::endl;
    const std::vector<unsigned int> shown = {1, 2, 3, 4, 5, 10, 20, 30, 40, 50};
    std::cout << "    episode:";
    for (auto e : shown) {
        std::cout << std::setw(7) << e;
    }
    std::cout << "    2 to 10" << std::endl;
    const std::vector<unsigned int> steps = {1, 2, 4, 8, 16};
    std::vector<double> early(steps.size(), 0);
    std::vector<std::vector<std::size_t>> greedy(steps.size());
    std::cout << std::fixed << std::setprecision(1);
    for (std::size_t i = 0; i < steps.size(); i++) {
        auto moves = moves_per_episode(steps[i], 50, 30, greedy[i]);
        // the mean moves of episodes 2 to 10 (the first is a random walk for every n)
        for (unsigned int e = 1; e < 10; e++) {
            early[i] += moves[e] / 9;
        }
        std::cout << "    n = " << std::setw(2) << steps[i] << ": ";
        for (auto e : shown) {
            std::cout << std::setw(7) << moves[e - 1];
        }
        std::cout << std::setw(11) << early[i] << std::endl;
    }

    // 3. the greedy paths after 50 episodes
    std::cout << std::endl << "The greedy path after 50 episodes, in moves (shortest 14):" << std::endl;
    bool reached = true;
    for (std::size_t i = 0; i < steps.size(); i++) {
        double mean = 0;
        std::size_t shortest = 1000;
        std::size_t longest = 0;
        for (auto m : greedy[i]) {
            mean += static_cast<double>(m) / static_cast<double>(greedy[i].size());
            shortest = std::min(shortest, m);
            longest = std::max(longest, m);
        }
        std::cout << "    n = " << std::setw(2) << steps[i] << ": mean " << mean
                  << ", from " << shortest << " to " << longest << std::endl;
        reached = reached && longest <= 30;
    }

    // a larger n must learn faster early on, and every greedy path must reach the goal
    bool learnt = reached && (early[2] < 0.75 * early[0]) && (early[3] < 0.75 * early[0]);
    return learnt ? EXIT_SUCCESS : EXIT_FAILURE;
}
