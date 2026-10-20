/**
 * Prioritized sweeping against Dyna-Q in the Dyna maze, at increasing sizes (Sutton & Barto, Example 8.4).
 *
 * The maze of Example 8.1 is made finer and finer: at scale k each cell becomes a block of
 * k x k cells, so the maze keeps its shape, but has 47 k^2 cells, and its shortest path takes 14 k moves.
 * Both agents play ε-greedy, and plan at most n = 5 times after each real move.
 * Dyna-Q plans with moves picked at random, prioritized sweeping with the moves whose
 * values would change the most. The program counts the updates of a value each one makes
 * until its greedy path is a good one (at most 20% longer than the shortest).
 *
 * The program succeeds if every run finds a good path, and if prioritized sweeping needs
 * fewer updates than Dyna-Q at every size.
 *
 * @see games/dyna_maze.hpp for the mazes
 */
#include "games/dyna_maze.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// set shortcuts to the game, to its relearn types, and to the two learners
using game_type     = games::dyna_maze;
using types         = games::types<game_type>;
using dyna_type     = relearn::dyna_q<types::state, types::action>;
using sweeping_type = relearn::prioritized_sweeping<types::state, types::action>;

/// @return the updates of Dyna-Q: each real move is one, followed by n imagined ones
std::size_t updates_of(const dyna_type & learner)
{
    return learner.steps() * (learner.planning_steps + 1);
}

/// @return the updates of prioritized sweeping: one for each move taken from the queue
std::size_t updates_of(const sweeping_type & learner)
{
    return learner.updates();
}

/**
 * @brief learn a maze until the greedy path is a good one
 * @param maze is the maze
 * @param scale is its scale (1 is the maze of Example 8.1)
 * @param learner is Dyna-Q or prioritized sweeping
 * @param memory receives the values learnt
 * @param seed seeds the agent
 * @return the updates made until then, or 0 if the greedy path never was a good one
 *
 * ε-greedy with ε = 0.1. After every episode the greedy policy walks the maze: a path at most
 * 20% longer than the shortest (14 moves at scale 1) is good.
 */
template <class learner_type>
std::size_t learn(game_type & maze,
                  int scale,
                  learner_type & learner,
                  types::policy & memory,
                  unsigned int seed)
{
    std::mt19937 prng(seed);
    std::mt19937 ties(seed + 500);
    const std::size_t good = static_cast<std::size_t>(1.2 * 14 * scale);
    // the ε-greedy agent, and the greedy one
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
    };
    auto greedy = [&](const game_type::state_type & s,
                      const std::vector<unsigned int> & legal) {
        return games::greedy<game_type>(memory, s, legal, ties);
    };
    for (unsigned int episode = 0; episode < 100000; episode++) {
        // play an episode, learning (and planning) after every move
        games::play_online(maze, agent, learner, memory, 1000000);
        // then walk the greedy path, without exploring
        auto walk = games::play(maze, greedy, good);
        if (walk.terminated) {
            return updates_of(learner);
        }
    }
    return 0;
}

/**
 * @brief draw the greedy path of @param memory through @param maze, with '*' on the cells it walks
 * @param prng breaks ties
 * @return the moves of the path
 */
std::size_t draw_path(game_type & maze,
                      const types::policy & memory,
                      std::mt19937 & prng)
{
    auto rows = maze.draw();
    auto at = maze.reset();
    std::size_t moves = 0;
    while (moves < 100 && !(at == maze.goal())) {
        if (!(at == maze.start())) {
            rows[static_cast<std::size_t>(at.y)][static_cast<std::size_t>(at.x)] = '*';
        }
        at = maze.step(games::greedy<game_type>(memory, at, maze.actions(at), prng)).next;
        moves++;
    }
    for (const auto & row : rows) {
        std::cout << "    ";
        for (char c : row) {
            std::cout << c << ' ';
        }
        std::cout << std::endl;
    }
    return moves;
}

/**
 * @brief compare the two learners, at scales 1 to 5
 * @param alpha is the learning rate of both
 * @param runs is the number of runs at each scale
 * @return true if every run found a good path, and prioritized sweeping always needed fewer updates
 */
bool compare(double alpha,
             unsigned int runs)
{
    std::cout << "    scale  cells  shortest     Dyna-Q   prioritized sweeping   fewer" << std::endl;
    bool fewer = true;
    for (int scale = 1; scale <= 5; scale++) {
        double dyna = 0;
        double sweeping = 0;
        for (unsigned int run = 1; run <= runs; run++) {
            // the same maze for both, n = 5, γ = 0.95, θ = 0.0001
            game_type maze(game_type::maze, 1000, scale);
            types::policy dyna_memory;
            types::policy sweeping_memory;
            dyna_type dyna_learner(alpha, 0.95, 5, run + 1000);
            sweeping_type sweeping_learner(alpha, 0.95, 5, 0.0001);
            auto by_dyna     = learn(maze, scale, dyna_learner, dyna_memory, run);
            auto by_sweeping = learn(maze, scale, sweeping_learner, sweeping_memory, run);
            if (by_dyna == 0 || by_sweeping == 0) {
                fewer = false;
            }
            dyna += static_cast<double>(by_dyna) / runs;
            sweeping += static_cast<double>(by_sweeping) / runs;
        }
        fewer = fewer && sweeping < dyna;
        std::cout << "    " << std::setw(5) << scale
                  << std::setw(7) << 47 * scale * scale
                  << std::setw(10) << 14 * scale
                  << std::setw(11) << std::setprecision(0) << dyna
                  << std::setw(23) << sweeping
                  << std::setw(7) << std::setprecision(1) << dyna / sweeping << "x" << std::endl;
    }
    return fewer;
}

int main()
{
    std::cout << std::fixed;
    // the maze at scale 2: each cell of Example 8.1 is a block of 2 x 2 cells
    std::cout << "the Dyna maze at scale 2 (18 x 12, shortest path 28 moves):" << std::endl;
    for (const auto & row : game_type(game_type::maze, 1000, 2).draw()) {
        std::cout << "    " << row << std::endl;
    }

    // Example 8.4: the updates until a good path, with α = 1 (the maze is deterministic)
    std::cout << std::endl << "updates until the greedy path is at most 20% longer than the shortest," << std::endl
              << "mean of 10 runs, n = 5, alpha = 1:" << std::endl;
    const bool learnt = compare(1.0, 10);
    // and with α = 0.5: each move taken from the queue goes only half way, and waits for
    // its next real move to be queued again
    std::cout << std::endl << "the same, with alpha = 0.5:" << std::endl;
    const bool learnt_half = compare(0.5, 10);

    // the greedy path of prioritized sweeping, after its first run at scale 2,
    // once it has planned with everything its model knows
    game_type maze(game_type::maze, 1000, 2);
    types::policy memory;
    sweeping_type learner(1.0, 0.95, 5, 0.0001);
    auto updates = learn(maze, 2, learner, memory, 1);
    // then it plans until its queue is empty, so that its values are those of its model
    while (learner.queued() > 0) {
        learner.plan(memory);
    }
    std::cout << std::endl << "the greedy path of prioritized sweeping at scale 2, after run 1 (" << updates
              << " updates)" << std::endl << "and planning until its queue is empty ("
              << learner.updates() - updates << " more updates):" << std::endl;
    std::mt19937 prng(1);
    auto moves = draw_path(maze, memory, prng);
    std::cout << "    " << moves << " moves, the shortest path takes 28" << std::endl;

    return learnt && learnt_half ? EXIT_SUCCESS : EXIT_FAILURE;
}
