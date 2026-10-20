/**
 * The Dyna maze (Sutton & Barto, Example 8.1) learnt with SARSA(λ) and Watkins's Q(λ) (Sections 12.7 and 12.10).
 *
 * 1. The first episode: with traces, every move made has a share of the goal's reward when the
 *    goal is reached, the later moves a larger share. With replacing traces, the best move of every
 *    cell visited is the one the agent last left it by, and following them leads to the goal without a loop.
 * 2. The moves per episode, averaged over 30 runs, of SARSA and Q-learning (λ = 0), and of SARSA(λ)
 *    and Q(λ) with λ = 0.9, with accumulating and with replacing traces.
 * 3. The greedy path learnt after 50 episodes, next to the shortest one (14 moves).
 *
 * Each episode is learnt once it has ended. The program succeeds if, with replacing traces,
 * SARSA(λ) and Q(λ) need less than a fifth of the moves of SARSA and Q-learning in episodes 2 to 10,
 * and if every greedy path reaches the goal in at most 30 moves.
 *
 * @see games/dyna_maze.hpp for the layout
 */
#include "games/dyna_maze.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// set shortcuts to the game and to its relearn types
using game_type = games::dyna_maze;
using types     = games::types<game_type>;
using sarsa_type = relearn::sarsa_lambda<types::state, types::action>;
using q_type     = relearn::watkins_q_lambda<types::state, types::action>;

/**
 * @brief the maze, with the greedy move drawn on every cell where some move has a value above 0
 * @param maze is the maze
 * @param memory is the learnt policy
 * @return one string per row
 */
std::vector<std::string> draw_greedy(const game_type & maze,
                                     const types::policy & memory)
{
    auto rows = maze.draw();
    const char arrows[] = {'^', '>', 'v', '<'};
    for (int y = 0; y < game_type::height; y++) {
        for (int x = 0; x < game_type::width; x++) {
            auto best = memory.best(types::state(games::position{x, y}));
            if (best.first && best.second > 0) {
                rows[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] = arrows[best.first->trait()];
            }
        }
    }
    return rows;
}

/**
 * @brief the first episode, learnt by one-step SARSA and by SARSA(λ) with replacing traces
 *
 * Every value is 0 until the goal is reached, so with the same seed both agents walk the same path.
 */
void first_episode()
{
    std::vector<std::vector<std::string>> drawn;
    std::vector<std::size_t> greedy_moves;
    std::size_t moves = 0;
    for (double lambda : {0.0, 0.9}) {
        std::mt19937 prng(32);
        game_type maze;
        types::policy memory;
        sarsa_type learner(0.1, 0.95, lambda, true);
        // the ε-greedy agent, with ε = 0.1 (ties between equal values are broken at random)
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        auto r = games::play(maze, agent, 100000);
        learner(r.episode, memory, r.terminated);
        moves = r.steps;
        drawn.push_back(draw_greedy(maze, memory));
        // then follow the greedy policy learnt from this one episode
        auto greedy = [&](const game_type::state_type & s,
                          const std::vector<unsigned int> & legal) {
            return games::greedy<game_type>(memory, s, legal, prng);
        };
        auto g = games::play(maze, greedy, 1000);
        greedy_moves.push_back(g.terminated ? g.steps : 0);
    }
    std::cout << "The first episode (" << moves << " moves), and the greedy move of every cell with a value" << std::endl;
    std::cout << "    SARSA        SARSA(λ = 0.9)" << std::endl;
    for (std::size_t y = 0; y < drawn[0].size(); y++) {
        std::cout << "    " << drawn[0][y] << "    " << drawn[1][y] << std::endl;
    }
    std::cout << "    following them, SARSA(λ)'s greedy policy reaches the goal in " << greedy_moves[1] << " moves" << std::endl;
}

/**
 * @brief the moves of each episode, averaged over many runs
 * @param learner is the learner, copied afresh for every run
 * @param episodes is the number of episodes of each run
 * @param runs is the number of runs, each a new agent learning from scratch
 * @param greedy_moves receives, for each run, the moves of the greedy policy learnt
 * @return the mean moves of each episode
 *
 * ε-greedy with ε = 0.1, α = 0.1, γ = 0.95, each episode learnt once it has ended.
 */
template <class learner_type>
std::vector<double> moves_per_episode(const learner_type & learner,
                                      unsigned int episodes,
                                      unsigned int runs,
                                      std::vector<std::size_t> & greedy_moves)
{
    std::vector<double> moves(episodes, 0);
    for (unsigned int run = 1; run <= runs; run++) {
        std::mt19937 prng(run);
        game_type maze;
        types::policy memory;
        learner_type fresh = learner;
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        // play each episode, then learn it
        for (unsigned int i = 0; i < episodes; i++) {
            auto r = games::play(maze, agent, 100000);
            fresh(r.episode, memory, r.terminated);
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

/**
 * @brief print one row of the table of moves per episode
 * @param name is the learner's name
 * @param moves are the mean moves of each episode
 * @param shown are the episodes to print
 * @return the mean moves of episodes 2 to 10
 */
double print_row(const std::string & name,
                 const std::vector<double> & moves,
                 const std::vector<unsigned int> & shown)
{
    // the mean moves of episodes 2 to 10 (the first is a random walk for every learner)
    double early = 0;
    for (unsigned int e = 1; e < 10; e++) {
        early += moves[e] / 9;
    }
    std::cout << "    " << std::setw(26) << std::left << name << std::right;
    for (auto e : shown) {
        std::cout << std::setw(7) << moves[e - 1];
    }
    std::cout << std::setw(10) << early << std::endl;
    return early;
}

int main()
{
    // 1. what the first episode teaches
    first_episode();

    // 2. moves per episode, over 30 runs
    std::cout << std::endl << "Moves per episode (shortest path 14 moves), 30 runs:" << std::endl;
    const std::vector<unsigned int> shown = {1, 2, 3, 5, 10, 20, 30, 50};
    std::cout << "    episode:                  ";
    for (auto e : shown) {
        std::cout << std::setw(7) << e;
    }
    std::cout << "   2 to 10" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    const std::vector<std::string> names = {"SARSA",
                                            "SARSA(0.9) accumulating",
                                            "SARSA(0.9) replacing",
                                            "Q-learning",
                                            "Q(0.9) accumulating",
                                            "Q(0.9) replacing"};
    std::vector<std::vector<std::size_t>> greedy(names.size());
    std::vector<double> early;
    early.push_back(print_row(names[0], moves_per_episode(sarsa_type(0.1, 0.95, 0.0), 50, 30, greedy[0]), shown));
    early.push_back(print_row(names[1], moves_per_episode(sarsa_type(0.1, 0.95, 0.9), 50, 30, greedy[1]), shown));
    early.push_back(print_row(names[2], moves_per_episode(sarsa_type(0.1, 0.95, 0.9, true), 50, 30, greedy[2]), shown));
    early.push_back(print_row(names[3], moves_per_episode(q_type(0.1, 0.95, 0.0), 50, 30, greedy[3]), shown));
    early.push_back(print_row(names[4], moves_per_episode(q_type(0.1, 0.95, 0.9), 50, 30, greedy[4]), shown));
    early.push_back(print_row(names[5], moves_per_episode(q_type(0.1, 0.95, 0.9, true), 50, 30, greedy[5]), shown));

    // 3. the greedy paths after 50 episodes
    std::cout << std::endl << "The greedy path after 50 episodes, in moves (shortest 14):" << std::endl;
    bool reached = true;
    for (std::size_t i = 0; i < names.size(); i++) {
        double mean = 0;
        std::size_t shortest = 1000;
        std::size_t longest = 0;
        for (auto m : greedy[i]) {
            mean += static_cast<double>(m) / static_cast<double>(greedy[i].size());
            shortest = std::min(shortest, m);
            longest = std::max(longest, m);
        }
        std::cout << "    " << std::setw(26) << std::left << names[i] << std::right
                  << "mean " << mean << ", from " << shortest << " to " << longest << std::endl;
        reached = reached && longest <= 30;
    }

    // replacing traces must learn much faster early on, and every greedy path must reach the goal
    bool learnt = reached && (early[2] < 0.2 * early[0]) && (early[5] < 0.2 * early[3]);
    return learnt ? EXIT_SUCCESS : EXIT_FAILURE;
}
