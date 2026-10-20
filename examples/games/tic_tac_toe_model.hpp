#ifndef RELEARN_GAMES_TIC_TAC_TOE_MODEL_HPP
#define RELEARN_GAMES_TIC_TAC_TOE_MODEL_HPP
/**
 * A model of tic-tac-toe for planning, e.g. with `relearn::mcts`.
 *
 * `games::tic_tac_toe` is the game from X's point of view, the opponent's answer hidden inside `step()`.
 * A planner has to play both sides in its head, so the model takes a board and a move, for whichever
 * player is to move, and gives back the board reached:
 *
 * - X moves first, so X is to move when both have marked as many cells, O otherwise
 * - the reward goes to the player who moved: +1 if the move wins, 0 otherwise
 *   (a move can never lose the game for the player who makes it)
 *
 * It also gives the perfect moves of a board, from the minimax values of `games::tic_tac_toe`.
 *
 * @see games/tic_tac_toe.hpp for the rules
 */
#include "tic_tac_toe.hpp"
#include <vector>

namespace games
{
/**
 * @struct tic_tac_toe_model
 * @brief tic-tac-toe as a model a planner can step: the board, and either player's move
 */
struct tic_tac_toe_model
{
    /// a state is the board
    using state_type  = board;
    /// an action is the empty cell to mark, 0 ... 8
    using action_type = unsigned int;

    /// @return the player to move on board @param b: 1 for X, 2 for O
    unsigned int player(const board & b) const
    {
        unsigned int x = 0;
        unsigned int o = 0;
        for (auto c : b.cells) {
            x += c == 1;
            o += c == 2;
        }
        return x == o ? 1 : 2;
    }

    /// @return the legal moves on board @param b: its empty cells, none if the game is over
    std::vector<action_type> actions(const board & b) const
    {
        if (b.winner() != 0) {
            return {};
        }
        return b.empty();
    }

    /// @brief the player to move marks cell @param a of board @param b
    /// @return the board reached, +1 if the move wins (0 otherwise), and whether the game ended
    step_result<board> step(const board & b,
                            const action_type & a) const
    {
        step_result<board> out;
        out.next = b;
        out.next.cells[a] = static_cast<unsigned char>(player(b));
        // the game ends with a line of three, which only the player who moved can have made
        bool won = out.next.winner() != 0;
        out.reward = won ? 1 : 0;
        out.terminated = won || out.next.full();
        return out;
    }
};

/**
 * @brief the perfect moves on board @param b, for the player to move
 * @param game works out the minimax values (and remembers them)
 * @return every empty cell of the best minimax value for the player to move, in order.
 *         The first one is the move of `games::tic_tac_toe`'s perfect opponent
 */
inline std::vector<unsigned int> perfect_moves(tic_tac_toe & game,
                                               const board & b)
{
    const unsigned char me = static_cast<unsigned char>(tic_tac_toe_model().player(b));
    const unsigned char next = me == 1 ? 2 : 1;
    // X takes the highest minimax value, O the lowest
    std::vector<unsigned int> best;
    int best_value = 0;
    for (unsigned int cell : b.empty()) {
        board after = b;
        after.cells[cell] = me;
        int v = game.minimax(after, next);
        bool better = me == 1 ? v > best_value : v < best_value;
        if (best.empty() || better) {
            best.assign(1, cell);
            best_value = v;
        }
        else if (v == best_value) {
            best.push_back(cell);
        }
    }
    return best;
}
} // namespace games
#endif
