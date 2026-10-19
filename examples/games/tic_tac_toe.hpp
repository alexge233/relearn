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
#include <array>
#include <random>
#include <unordered_map>
#include <vector>

namespace games
{
/// a tic-tac-toe board: 0 empty, 1 X (the agent), 2 O (the opponent)
struct board
{
    std::array<unsigned char, 9> cells{};

    bool operator==(const board & rhs) const { return cells == rhs.cells; }

    /// @return 1 if X has three in a row, 2 if O has, 0 otherwise
    unsigned char winner() const
    {
        static const int lines[8][3] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
                                        {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};
        for (const auto & l : lines) {
            if (cells[l[0]] != 0 && cells[l[0]] == cells[l[1]] && cells[l[1]] == cells[l[2]]) {
                return cells[l[0]];
            }
        }
        return 0;
    }
    bool full() const
    {
        for (auto c : cells) if (c == 0) return false;
        return true;
    }
    std::vector<unsigned int> empty() const
    {
        std::vector<unsigned int> free;
        for (unsigned int i = 0; i < 9; i++) if (cells[i] == 0) free.push_back(i);
        return free;
    }
    /// base-3 number of the board, unique for each board
    unsigned int code() const
    {
        unsigned int c = 0;
        for (auto v : cells) c = c * 3 + v;
        return c;
    }
};

class tic_tac_toe
{
public:
    using state_type  = board;
    using action_type = unsigned int;
    enum opponent_type { random, perfect };

    explicit tic_tac_toe(opponent_type opponent = random, unsigned int seed = 0)
    : opponent_(opponent), prng_(seed)
    {}

    state_type reset()
    {
        board_ = board();
        return board_;
    }

    /// put the game in position @param b (for tests)
    void place(const board & b)
    {
        board_ = b;
    }

    /// mark cell @param a with X, then let the opponent answer
    step_result<state_type> step(const action_type & a)
    {
        step_result<state_type> out;
        board_.cells[a] = 1;
        if (finished(out)) return out;
        board_.cells[opponent_move()] = 2;
        finished(out);
        return out;
    }

    std::vector<action_type> actions(const state_type & s) const
    {
        return s.empty();
    }

    /// minimax value of @param b with @param player to move: +1 X wins, 0 draw, -1 O wins
    int minimax(const board & b, unsigned char player)
    {
        unsigned int key = b.code() * 3 + player;
        auto it = memo_.find(key);
        if (it != memo_.end()) return it->second;
        int value;
        unsigned char w = b.winner();
        if (w != 0) {
            value = w == 1 ? 1 : -1;
        }
        else if (b.full()) {
            value = 0;
        }
        else {
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
    /// fill in @param out if the game is over: @return true if it is
    bool finished(step_result<state_type> & out) const
    {
        out.next = board_;
        unsigned char w = board_.winner();
        if (w != 0 || board_.full()) {
            out.terminated = true;
            out.reward = w == 1 ? 1 : (w == 2 ? -1 : 0);
            return true;
        }
        return false;
    }

    unsigned int opponent_move()
    {
        auto free = board_.empty();
        if (opponent_ == random) {
            std::uniform_int_distribution<std::size_t> any(0, free.size() - 1);
            return free[any(prng_)];
        }
        // perfect: the first move with the lowest minimax value for X
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

    opponent_type opponent_;
    std::mt19937 prng_;
    board board_;
    std::unordered_map<unsigned int, int> memo_;
};
} // namespace games

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
