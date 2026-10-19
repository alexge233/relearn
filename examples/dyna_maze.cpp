/**
 * The Dyna maze (Sutton & Barto, Example 8.1), the blocking maze (Example 8.2) and
 * the shortcut maze (Example 8.3), learnt with `dyna_q`.
 *
 * 1. Figure 8.2: the moves per episode in the Dyna maze, with n = 0 (Q-learning),
 *    5 and 50 imagined steps after each real one. The more the agent plans, the
 *    fewer episodes it needs to learn its way to the goal.
 * 2. Figure 8.4: in the blocking maze the short way closes after 1000 moves.
 *    Dyna-Q+ finds the new way quickly; Dyna-Q, whose model still believes in the
 *    old way, takes much longer.
 * 3. Figure 8.5: in the shortcut maze a shorter way opens after 3000 moves.
 *    Dyna-Q never finds it, since its model says the old way works fine;
 *    Dyna-Q+'s bonus sends it to check the moves it has not made for a long time.
 *
 * The program succeeds if planning learns the maze in fewer episodes, and if
 * Dyna-Q+ reaches the goal more often than Dyna-Q in both changing mazes.
 *
 * @see games/dyna_maze.hpp for the layouts
 */
#include "games/dyna_maze.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

// set shortcuts to the game, to its relearn types, and to the learner
using game_type    = games::dyna_maze;
using types        = games::types<game_type>;
using learner_type = relearn::dyna_q<types::state, types::action>;

/**
 * @brief the moves of each episode in the Dyna maze, averaged over many runs (Figure 8.2)
 * @param planning_steps is n, the imagined steps after each real one
 * @param episodes is the number of episodes of each run
 * @param runs is the number of runs, each a new agent learning from scratch
 * @param greedy_moves receives, for each run, the moves of the greedy policy learnt
 * @return the mean moves of each episode
 *
 * The settings of Sutton & Barto: ε-greedy with ε = 0.1, α = 0.1, γ = 0.95.
 */
std::vector<double> moves_per_episode(unsigned int planning_steps,
                                      unsigned int episodes,
                                      unsigned int runs,
                                      std::vector<std::size_t> & greedy_moves)
{
    std::vector<double> moves(episodes, 0);
    for (unsigned int run = 1; run <= runs; run++) {
        std::mt19937 prng(run);
        game_type maze;
        types::policy memory;
        learner_type learner(0.1, 0.95, planning_steps, run + 1000);
        // the ε-greedy agent, with ε = 0.1
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        // play the episodes, learning (and planning) after every move
        for (unsigned int i = 0; i < episodes; i++) {
            auto r = games::play_online(maze, agent, learner, memory, 100000);
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
 * @brief the goals reached so far, every `report` moves, in a maze that changes (Figures 8.4, 8.5)
 * @param layout is the blocking or the shortcut maze
 * @param change is the move at which the maze changes
 * @param total is the number of moves of each run
 * @param report is the number of moves between two counts
 * @param kappa is κ, Dyna-Q+'s bonus: 0 for Dyna-Q
 * @param runs is the number of runs, each a new agent learning from scratch
 * @return the mean goals reached so far, after each `report` moves
 *
 * ε-greedy with ε = 0.1, α = 1 (the maze is deterministic), γ = 0.95, n = 50.
 */
std::vector<double> goals_so_far(game_type::layout_type layout,
                                 unsigned long change,
                                 unsigned long total,
                                 unsigned long report,
                                 double kappa,
                                 unsigned int runs)
{
    std::vector<double> goals(total / report, 0);
    for (unsigned int run = 1; run <= runs; run++) {
        std::mt19937 prng(run);
        game_type maze(layout, change);
        types::policy memory;
        learner_type learner(1.0, 0.95, 50, run + 1000);
        learner.kappa = kappa;
        // Dyna-Q+ needs to know every move of a state, to imagine the ones it has not tried
        learner.legal_actions = [&maze](const types::state & s) {
            std::vector<types::action> moves;
            for (const auto & a : maze.actions(s.trait())) {
                moves.push_back(types::action(a));
            }
            return moves;
        };
        // the ε-greedy agent, with ε = 0.1
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        // play until the run's moves are used up, counting the goals reached
        double reached = 0;
        std::size_t next_report = 0;
        while (maze.moves() < total) {
            auto r = games::play_online(maze, agent, learner, memory, total - maze.moves());
            // the counts the episode went past before it ended: its goal came after them,
            // since the maze rewards only the move onto G, which is the episode's last move
            while (next_report < goals.size() && maze.moves() > (next_report + 1) * report) {
                goals[next_report] += reached / runs;
                next_report++;
            }
            // then count its goal, which includes a count at the very move it ended on
            reached += r.total_reward;
            while (next_report < goals.size() && maze.moves() >= (next_report + 1) * report) {
                goals[next_report] += reached / runs;
                next_report++;
            }
        }
    }
    return goals;
}

/**
 * @brief print a maze before and after it changes, side by side
 * @param layout is the maze
 * @param change is the move at which it changes
 */
void draw_change(game_type::layout_type layout,
                 unsigned long change)
{
    game_type before(layout, change);
    game_type after(layout, 0);
    std::cout << "    before       after" << std::endl;
    auto rows_before = before.draw();
    auto rows_after  = after.draw();
    for (std::size_t y = 0; y < rows_before.size(); y++) {
        std::cout << "    " << rows_before[y] << "    " << rows_after[y] << std::endl;
    }
}

/**
 * @brief print the goals reached so far by Dyna-Q and Dyna-Q+, side by side
 * @param report is the number of moves between two counts
 * @param dyna and @param plus are the counts of Dyna-Q and Dyna-Q+
 */
void print_goals(unsigned long report,
                 const std::vector<double> & dyna,
                 const std::vector<double> & plus)
{
    std::cout << "    moves   Dyna-Q   Dyna-Q+" << std::endl;
    for (std::size_t i = 0; i < dyna.size(); i++) {
        std::cout << "    " << std::setw(5) << (i + 1) * report
                  << std::setw(9) << dyna[i]
                  << std::setw(10) << plus[i] << std::endl;
    }
}

int main()
{
    std::cout << std::fixed << std::setprecision(1);

    // 1. Figure 8.2: moves per episode, with n = 0, 5 and 50, over 30 runs
    std::cout << "Dyna maze (shortest path 14 moves), moves per episode, 30 runs:" << std::endl;
    for (const auto & row : game_type().draw()) {
        std::cout << "    " << row << std::endl;
    }
    const std::vector<unsigned int> shown = {1, 2, 3, 4, 5, 10, 20, 30, 40, 50};
    std::cout << "    episode:";
    for (auto e : shown) {
        std::cout << std::setw(7) << e;
    }
    std::cout << std::endl;
    std::vector<double> early(3, 0);
    const std::vector<unsigned int> planning = {0, 5, 50};
    std::vector<std::vector<std::size_t>> greedy(planning.size());
    for (std::size_t i = 0; i < planning.size(); i++) {
        auto moves = moves_per_episode(planning[i], 50, 30, greedy[i]);
        std::cout << "    n = " << std::setw(2) << planning[i] << ": ";
        for (auto e : shown) {
            std::cout << std::setw(7) << moves[e - 1];
        }
        std::cout << std::endl;
        // the mean moves of episodes 2 to 10 (the first is a random walk for every n)
        for (unsigned int e = 1; e < 10; e++) {
            early[i] += moves[e] / 9;
        }
    }
    // the greedy policy after 50 episodes: Dyna-Q only plans with moves it has made,
    // so a run that never made the moves of a shortest path cannot know it is shorter
    std::cout << "    greedy policy after 50 episodes, moves (runs):" << std::endl;
    for (std::size_t i = 0; i < planning.size(); i++) {
        std::map<std::size_t, unsigned int> lengths;
        for (auto moves : greedy[i]) {
            lengths[moves]++;
        }
        std::cout << "    n = " << std::setw(2) << planning[i] << ":";
        for (const auto & length : lengths) {
            std::cout << "  " << length.first << " (" << length.second << ")";
        }
        std::cout << std::endl;
    }

    // 2. Figure 8.4: the blocking maze, the short way closes after 1000 moves
    std::cout << std::endl << "blocking maze, the wall moves after 1000 moves, goals reached (20 runs):" << std::endl;
    draw_change(game_type::blocking, 1000);
    auto blocked      = goals_so_far(game_type::blocking, 1000, 3000, 500, 0, 20);
    auto blocked_plus = goals_so_far(game_type::blocking, 1000, 3000, 500, 0.001, 20);
    print_goals(500, blocked, blocked_plus);

    // 3. Figure 8.5: the shortcut maze, a shorter way opens after 3000 moves
    std::cout << std::endl << "shortcut maze, a shortcut opens after 3000 moves, goals reached (20 runs):" << std::endl;
    draw_change(game_type::shortcut, 3000);
    auto shortcut      = goals_so_far(game_type::shortcut, 3000, 6000, 1000, 0, 20);
    auto shortcut_plus = goals_so_far(game_type::shortcut, 3000, 6000, 1000, 0.001, 20);
    print_goals(1000, shortcut, shortcut_plus);

    // planning learns the maze in fewer episodes, and Dyna-Q+ wins in both changing mazes
    const bool learnt = early[2] < early[1] && early[1] < early[0] &&
                        blocked_plus.back() > blocked.back() &&
                        shortcut_plus.back() > shortcut.back();
    return learnt ? EXIT_SUCCESS : EXIT_FAILURE;
}
