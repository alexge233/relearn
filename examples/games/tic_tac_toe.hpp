#ifndef RELEARN_GAMES_TIC_TAC_TOE_HPP
#define RELEARN_GAMES_TIC_TAC_TOE_HPP
/**
 * Tic-tac-toe against a built-in opponent (Sutton & Barto, Section 1.5).
 *
 * - The agent plays X and moves first; the opponent (O) answers inside `step()`,
 *   so from the agent's point of view the opponent is part of the game.
 * - The state is the board, the action the empty cell 0 ... 8 to mark:
 *
 *       0 | 1 | 2
 *       3 | 4 | 5
 *       6 | 7 | 8
 *
 * - Rewards: +1 win, -1 loss, 0 draw, at the end of the game.
 * - Opponents: `random` plays any empty cell; `perfect` plays minimax and never
 *   loses, so the best the agent can do against it is a draw.
 */
#include "game.hpp"
#include <algorithm>
#include <array>
#include <random>
#include <unordered_map>
#include <vector>

namespace games
{
/**
 * @struct board
 * @brief a tic-tac-toe board: each cell is 0 empty, 1 X (the agent) or 2 O (the opponent)
 */
struct board
{
    std::array<unsigned char, 9> cells{};

    /// @brief two boards are equal when every cell is the same
    bool operator==(const board & rhs) const
    {
        return cells == rhs.cells;
    }

    /// @return 1 if X has three in a row, 2 if O has, 0 otherwise
    unsigned char winner() const
    {
        // the 8 lines of three: 3 rows, 3 columns and 2 diagonals
        static const int lines[8][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
                                        {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};
        for (const auto & l : lines) {
            if (cells[l[0]] != 0 &&
                cells[l[0]] == cells[l[1]] &&
                cells[l[1]] == cells[l[2]]) {
                return cells[l[0]];
            }
        }
        return 0;
    }

    /// @return true if no cell is empty
    bool full() const
    {
        for (auto c : cells) {
            if (c == 0) {
                return false;
            }
        }
        return true;
    }

    /// @return the empty cells, in order
    std::vector<unsigned int> empty() const
    {
        std::vector<unsigned int> free;
        for (unsigned int i = 0; i < 9; i++) {
            if (cells[i] == 0) {
                free.push_back(i);
            }
        }
        return free;
    }

    /// @return the board as a base-3 number, unique for each board
    unsigned int code() const
    {
        unsigned int c = 0;
        for (auto v : cells) {
            c = c * 3 + v;
        }
        return c;
    }
};

/**
 * @class tic_tac_toe
 * @brief tic-tac-toe, the agent playing X against a random or a perfect opponent
 */
class tic_tac_toe
{
public:
    /// a state is the board
    using state_type  = board;
    /// an action is the empty cell to mark, 0 ... 8
    using action_type = unsigned int;
    /// the two opponents
    enum opponent_type
    {
        random,
        perfect
    };

    /**
     * @brief create the game
     * @param opponent is the opponent to play against
     * @param seed seeds the moves of the random opponent
     */
    explicit tic_tac_toe(opponent_type opponent = random,
                         unsigned int seed = 0)
    : opponent_(opponent),
      prng_(seed)
    {}

    /// @brief start a new game @return the empty board
    state_type reset()
    {
        board_ = board();
        return board_;
    }

    /// @brief put the game in position @param b (for tests)
    void place(const board & b)
    {
        board_ = b;
    }

    /// @brief mark cell @param a with X, then let the opponent answer
    /// @return the board after both moves, the reward and whether the game ended
    step_result<state_type> step(const action_type & a)
    {
        step_result<state_type> out;
        // the agent's move
        board_.cells[a] = 1;
        if (finished(out)) {
            return out;
        }
        // the opponent's answer
        board_.cells[opponent_move()] = 2;
        finished(out);
        return out;
    }

    /// @return the legal moves in @param s: its empty cells
    std::vector<action_type> actions(const state_type & s) const
    {
        return s.empty();
    }

    /**
     * @brief the minimax value of a board, from X's point of view
     * @param b is the board
     * @param player is the player to move: 1 for X, 2 for O
     * @return +1 if X wins with best play from both sides, 0 for a draw, -1 if O wins
     */
    int minimax(const board & b,
                unsigned char player)
    {
        // a board is solved once, then its value is remembered
        unsigned int key = b.code() * 3 + player;
        auto it = memo_.find(key);
        if (it != memo_.end()) {
            return it->second;
        }
        int value;
        unsigned char w = b.winner();
        if (w != 0) {
            // the game is won
            value = w == 1 ? 1 : -1;
        }
        else if (b.full()) {
            // the game is drawn
            value = 0;
        }
        else {
            // try every empty cell: X takes the highest value, O the lowest
            value = player == 1 ? -2 : 2;
            for (unsigned int cell : b.empty()) {
                board next = b;
                next.cells[cell] = player;
                int v = minimax(next, player == 1 ? 2 : 1);
                value = player == 1 ? std::max(value, v) : std::min(value, v);
            }
        }
        memo_[key] = value;
        return value;
    }

private:
    /// @brief fill in @param out with the board, and the reward if the game is over
    /// @return true if the game is over
    bool finished(step_result<state_type> & out) const
    {
        out.next = board_;
        unsigned char w = board_.winner();
        if (w != 0 || board_.full()) {
            out.terminated = true;
            // +1 if X won, -1 if O won, 0 for a draw
            out.reward = w == 1 ? 1 : (w == 2 ? -1 : 0);
            return true;
        }
        return false;
    }

    /// @return the cell the opponent marks
    unsigned int opponent_move()
    {
        auto free = board_.empty();
        // the random opponent marks any empty cell
        if (opponent_ == random) {
            std::uniform_int_distribution<std::size_t> any(0, free.size() - 1);
            return free[any(prng_)];
        }
        // the perfect opponent marks the first cell with the lowest minimax value for X
        unsigned int best = free.front();
        int best_value = 2;
        for (unsigned int cell : free) {
            board next = board_;
            next.cells[cell] = 2;
            int v = minimax(next, 1);
            if (v < best_value) {
                best_value = v;
                best = cell;
            }
        }
        return best;
    }

    // which opponent plays O
    opponent_type opponent_;
    // the moves of the random opponent
    std::mt19937 prng_;
    // the current board
    board board_;
    // minimax values already computed, by board and player to move
    std::unordered_map<unsigned int, int> memo_;
};
} // namespace games

/**
 * Hash specialisation in the STD namespace for struct board.
 * It is **required** because relearn stores states in unordered maps,
 * which use hashing functions for the classes they store.
 */
namespace std
{
template <> struct hash<games::board>
{
    std::size_t operator()(const games::board & b) const
    {
        return std::hash<unsigned int>()(b.code());
    }
};
}
#endif
