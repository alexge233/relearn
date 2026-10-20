/**
 * Tic-tac-toe (Sutton & Barto, Section 1.5) played by Monte Carlo Tree Search
 * (Section 8.11), with UCT as the tree policy and random rollouts.
 *
 * MCTS learns nothing that lasts: before each move it plays thousands of games
 * in its head, from the current board, and then plays the move it tried most.
 *
 * 1. The search from the empty board: how often each first move was tried, and its value.
 * 2. Four positions, where it must win at once, or block the opponent's win.
 * 3. Against a perfect (minimax) opponent and a random one, as X and as O,
 *    with more and more iterations per move.
 *
 * The program succeeds if, with 5000 iterations per move, it never loses a game
 * to either opponent, and beats the random one in at least 95% of its games as X
 * and 80% as O.
 *
 * @see games/tic_tac_toe.hpp for the rules, and games/tic_tac_toe_model.hpp for the model MCTS plans with
 */
#include "games/tic_tac_toe_model.hpp"
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// set shortcuts to the model, the planner, and a player (who picks a cell on a board)
using model_type   = games::tic_tac_toe_model;
using planner_type = relearn::mcts<model_type>;
using player_type  = std::function<unsigned int(const games::board &)>;

/**
 * @brief play one game of tic-tac-toe between @param x and @param o
 * @return 1 if X wins, 2 if O wins, 0 for a draw
 */
unsigned int match(const player_type & x,
                   const player_type & o)
{
    model_type model;
    games::board b;
    while (true) {
        // the player to move marks a cell
        unsigned int mover = model.player(b);
        auto out = model.step(b, mover == 1 ? x(b) : o(b));
        b = out.next;
        if (out.terminated) {
            // the only reward is the winner's
            return out.reward > 0 ? mover : 0;
        }
    }
}

/// @return the mark of cell @param c of board @param b: X, O, or a dot when it is empty
char mark(const games::board & b,
          unsigned int c)
{
    return b.cells[c] == 1 ? 'X' : (b.cells[c] == 2 ? 'O' : '.');
}

/**
 * @brief the results of @param games_played games as X and as O, against @param opponent
 * @param agent is the MCTS player
 * @return the number of {wins, draws, losses} as X, then as O
 */
std::vector<int> results(const player_type & agent,
                         const player_type & opponent,
                         int games_played)
{
    std::vector<int> wdl(6, 0);
    for (int i = 0; i < games_played; i++) {
        // as X: a win is 1, a loss 2
        auto as_x = match(agent, opponent);
        wdl[as_x == 1 ? 0 : (as_x == 0 ? 1 : 2)]++;
        // as O: a win is 2, a loss 1
        auto as_o = match(opponent, agent);
        wdl[as_o == 2 ? 3 : (as_o == 0 ? 4 : 5)]++;
    }
    return wdl;
}

/// @brief print the wins, draws and losses @param wdl from index @param from, as "W/D/L"
void print_wdl(const std::vector<int> & wdl,
               std::size_t from)
{
    std::cout << std::setw(4) << wdl[from] << " /" << std::setw(4) << wdl[from + 1]
              << " /" << std::setw(4) << wdl[from + 2];
}

int main()
{
    // c = 2√2: UCB1's √2 is for returns between 0 and 1, and these go from -1 to 1
    const double c = 2 * std::sqrt(2.0);
    const unsigned int strongest = 5000;
    bool ok = true;

    // 1. the search from the empty board
    planner_type planner(model_type(), strongest, c, 1);
    games::board empty;
    planner.search(empty);
    const auto & tree = planner.tree();
    std::cout << "1. The search from the empty board, " << strongest << " iterations (UCT, c = 2 sqrt(2) = "
              << std::setprecision(3) << c << ", random rollouts)" << std::endl;
    std::cout << "   the visits of each first move, and its value for X:" << std::endl;
    std::vector<std::size_t> visits(9, 0);
    std::vector<double> values(9, 0);
    for (std::size_t i = 0; i < tree[0].children.size(); i++) {
        visits[tree[0].actions[i]] = tree[tree[0].children[i]].visits;
        values[tree[0].actions[i]] = tree[tree[0].children[i]].value();
    }
    std::cout << std::fixed << std::setprecision(2);
    for (unsigned int row = 0; row < 3; row++) {
        std::cout << "     ";
        for (unsigned int col = 0; col < 3; col++) {
            std::cout << std::setw(5) << visits[row * 3 + col];
        }
        std::cout << "        ";
        for (unsigned int col = 0; col < 3; col++) {
            std::cout << std::setw(6) << values[row * 3 + col];
        }
        std::cout << std::endl;
    }
    std::cout << "   it plays cell " << planner.best_action() << ", the most visited" << std::endl;

    // 2. win at once, or block
    struct puzzle
    {
        std::string what;
        std::vector<unsigned int> x;
        std::vector<unsigned int> o;
        unsigned int answer;
    };
    std::vector<puzzle> puzzles = {{"X to move, wins at 2 (rather than block 5)", {0, 1}, {3, 4}, 2},
                                   {"X to move, blocks at 6 (and then forks)", {0, 8}, {2, 4}, 6},
                                   {"O to move, blocks at 2", {0, 1}, {4}, 2},
                                   {"O to move, wins at 5 (rather than block 2)", {0, 1, 8}, {3, 4}, 5}};
    std::cout << std::endl << "2. Positions, " << strongest << " iterations:" << std::endl;
    for (const auto & q : puzzles) {
        games::board b;
        for (auto cell : q.x) {
            b.cells[cell] = 1;
        }
        for (auto cell : q.o) {
            b.cells[cell] = 2;
        }
        unsigned int move = planner(b);
        for (unsigned int row = 0; row < 3; row++) {
            std::cout << "     " << mark(b, row * 3) << mark(b, row * 3 + 1) << mark(b, row * 3 + 2);
            if (row == 1) {
                std::cout << "   " << q.what << ": it plays " << move
                          << (move == q.answer ? "" : "  WRONG");
            }
            std::cout << std::endl;
        }
        ok = ok && move == q.answer;
    }

    // 3. against a perfect and a random opponent
    games::tic_tac_toe solver;
    std::mt19937 prng(3);
    // the perfect opponent: one of the best minimax moves, at random (so that the games differ)
    player_type perfect = [&](const games::board & b) {
        auto best = games::perfect_moves(solver, b);
        std::uniform_int_distribution<std::size_t> pick(0, best.size() - 1);
        return best[pick(prng)];
    };
    // the random opponent: any empty cell
    player_type random = [&](const games::board & b) {
        auto free = b.empty();
        std::uniform_int_distribution<std::size_t> pick(0, free.size() - 1);
        return free[pick(prng)];
    };
    const int games_played = 100;
    std::cout << std::endl << "3. " << games_played << " games as X and " << games_played
              << " as O against each opponent (wins / draws / losses):" << std::endl;
    std::cout << "   iterations     perfect, as X      perfect, as O       random, as X       random, as O" << std::endl;
    for (unsigned int iterations : {10u, 100u, 1000u, strongest}) {
        planner_type agent_planner(model_type(), iterations, c, 2);
        player_type agent = [&](const games::board & b) {
            return agent_planner(b);
        };
        auto vs_perfect = results(agent, perfect, games_played);
        auto vs_random  = results(agent, random, games_played);
        std::cout << std::setw(13) << iterations << "   ";
        print_wdl(vs_perfect, 0);
        std::cout << "   ";
        print_wdl(vs_perfect, 3);
        std::cout << "   ";
        print_wdl(vs_random, 0);
        std::cout << "   ";
        print_wdl(vs_random, 3);
        std::cout << std::endl;
        // the strongest search must never lose, and must beat the random opponent
        if (iterations == strongest) {
            ok = ok && vs_perfect[2] == 0 && vs_perfect[5] == 0;
            ok = ok && vs_random[2] == 0 && vs_random[5] == 0;
            ok = ok && vs_random[0] >= 95 && vs_random[3] >= 80;
        }
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
