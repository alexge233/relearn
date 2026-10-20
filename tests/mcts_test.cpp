// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/tic_tac_toe_model.hpp"
#include <functional>
#include <vector>
#define CATCH_CONFIG_MAIN
#include <string>
#include "catch.hpp"

/**
 * Tests of Monte Carlo Tree Search (Sutton & Barto, Section 8.11), with UCT:
 * - tiny hand-made games, where the visits and values of every node are worked out by hand:
 *   UCT's choices, discounting, rollouts cut short, the sign of the values with two players,
 *   and the final move by visit count
 * - tic-tac-toe positions where it must win at once, or block a win
 * - tic-tac-toe against a perfect (minimax) opponent, which it never loses to,
 *   and against a random one, which it almost always beats
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using step_type = games::step_result<int>;

/// @return the outcome of a step: the state @param next, the @param reward, and whether it @param ends
step_type outcome(int next,
                  double reward,
                  bool ends)
{
    step_type out;
    out.next = next;
    out.reward = reward;
    out.terminated = ends;
    return out;
}

/**
 * @struct one_move
 * @brief a one-player game of a single move: action 0 earns 0, action 1 earns 1, action 2 earns 0.5
 */
struct one_move
{
    using state_type  = int;
    using action_type = int;

    std::vector<int> actions(const int &) const
    {
        return {0, 1, 2};
    }

    step_type step(const int &,
                   const int & a) const
    {
        const double rewards[3] = {0, 1, 0.5};
        return outcome(a + 1, rewards[a], true);
    }

    unsigned int player(const int &) const
    {
        return 0;
    }
};

/**
 * @struct chain
 * @brief a one-player chain 0 -> 1 -> 2 -> 3 with a single action, earning 0, 0, then 1 at the end
 */
struct chain
{
    using state_type  = int;
    using action_type = int;

    std::vector<int> actions(const int &) const
    {
        return {0};
    }

    step_type step(const int & s,
                   const int &) const
    {
        return outcome(s + 1, s == 2 ? 1 : 0, s == 2);
    }

    unsigned int player(const int &) const
    {
        return 0;
    }
};

/**
 * @struct two_player_chain
 * @brief a two-player chain 0 -> 1 -> 2 -> 3 with a single action: player 0 moves, then player 1,
 *        then player 0 again, whose move ends the game and earns player 0 a reward of 1
 */
struct two_player_chain
{
    using state_type  = int;
    using action_type = int;

    std::vector<int> actions(const int &) const
    {
        return {0};
    }

    step_type step(const int & s,
                   const int &) const
    {
        return outcome(s + 1, s == 2 ? 1 : 0, s == 2);
    }

    unsigned int player(const int & s) const
    {
        return s == 1 ? 1 : 0;
    }
};

/**
 * @struct safe_or_deep
 * @brief a one-player game: in state 0, action 0 (safe) ends the game with 0.6, whereas action 1 (deep)
 *        earns 0 and leads to state 2, whose single action ends the game with 1
 */
struct safe_or_deep
{
    using state_type  = int;
    using action_type = int;

    std::vector<int> actions(const int & s) const
    {
        if (s == 0) {
            return {0, 1};
        }
        return {0};
    }

    step_type step(const int & s,
                   const int & a) const
    {
        if (s == 0 && a == 0) {
            return outcome(1, 0.6, true);
        }
        if (s == 0) {
            return outcome(2, 0, false);
        }
        return outcome(3, 1, true);
    }

    unsigned int player(const int &) const
    {
        return 0;
    }
};

/**
 * @struct two_player_tree
 * @brief a two-player game of two moves. Player 0 moves first, left (0) or right (1):
 *        - left earns player 0 a reward of 0.5, then player 1's two moves each earn player 1 0.25
 *        - right earns player 0 nothing, then player 1's two moves each earn player 1 1
 *        From player 0's point of view, left is worth 0.5 - 0.25 = 0.25 and right 0 - 1 = -1.
 */
struct two_player_tree
{
    using state_type  = int;
    using action_type = int;

    std::vector<int> actions(const int &) const
    {
        return {0, 1};
    }

    step_type step(const int & s,
                   const int & a) const
    {
        // state 0: player 0 moves to state 1 (left) or 2 (right)
        if (s == 0) {
            return a == 0 ? outcome(1, 0.5, false) : outcome(2, 0, false);
        }
        // states 1 and 2: player 1 moves, and the game ends
        return outcome(3, s == 1 ? 0.25 : 1, true);
    }

    unsigned int player(const int & s) const
    {
        return s == 1 || s == 2 ? 1 : 0;
    }
};

using planner = relearn::mcts<games::tic_tac_toe_model>;
using player_type = std::function<unsigned int(const games::board &)>;

/// the exploration constant c = 2√2: UCB1's √2 is for returns between 0 and 1, and tic-tac-toe's go from -1 to 1, twice as wide
const double exploration = 2 * std::sqrt(2.0);

/// @return the board with X on the cells @param x and O on the cells @param o
games::board position(std::vector<unsigned int> x,
                      std::vector<unsigned int> o)
{
    games::board b;
    for (auto c : x) {
        b.cells[c] = 1;
    }
    for (auto c : o) {
        b.cells[c] = 2;
    }
    return b;
}

/**
 * @brief play one game of tic-tac-toe between @param x and @param o
 * @return 1 if X wins, 2 if O wins, 0 for a draw
 */
unsigned int match(const player_type & x,
                   const player_type & o)
{
    games::tic_tac_toe_model model;
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

/// @return a perfect player: one of the best minimax moves, picked at random with @param prng
player_type perfect_player(games::tic_tac_toe & solver,
                           std::mt19937 & prng)
{
    return [&](const games::board & b) {
        auto best = games::perfect_moves(solver, b);
        std::uniform_int_distribution<std::size_t> pick(0, best.size() - 1);
        return best[pick(prng)];
    };
}

/// @return a random player: any empty cell, picked with @param prng
player_type random_player(std::mt19937 & prng)
{
    return [&](const games::board & b) {
        auto free = b.empty();
        std::uniform_int_distribution<std::size_t> pick(0, free.size() - 1);
        return free[pick(prng)];
    };
}

/// @return the MCTS player, searching with @param p
player_type mcts_player(planner & p)
{
    return [&](const games::board & b) {
        return p(b);
    };
}
} // namespace

/// verifies that UCT uses ln N of the parent, over many iterations, against the formula worked out here
TEST_CASE("mcts: UCT's visits follow its formula exactly", "[mcts]")
{
    // On the game of one move every return is the action's reward (0, 1, 0.5), so UCT's choices can be
    // replayed here: iterations 1 to 3 expand the actions in order, and iteration k + 1 (k = 3, 4, ...)
    // picks the first action of highest Q + c * sqrt(ln k / N), k being the root's visits so far
    const double rewards[3] = {0, 1, 0.5};
    for (unsigned int iterations = 4; iterations <= 300; iterations++) {
        unsigned int visits[3] = {1, 1, 1};
        for (unsigned int k = 3; k < iterations; k++) {
            unsigned int best = 0;
            double best_score = 0;
            for (unsigned int a = 0; a < 3; a++) {
                double score = rewards[a] + 1.0 * std::sqrt(std::log(static_cast<double>(k)) / visits[a]);
                if (a == 0 || score > best_score) {
                    best = a;
                    best_score = score;
                }
            }
            visits[best]++;
        }
        relearn::mcts<one_move> search(one_move(), iterations, 1.0);
        search.search(0);
        const auto & tree = search.tree();
        for (unsigned int a = 0; a < 3; a++) {
            REQUIRE(tree[tree[0].children[a]].visits == visits[a]);
        }
    }
}

/// verifies UCT's choices, worked out by hand, on a game of one move
TEST_CASE("mcts: UCT on a game of one move", "[mcts]")
{
    // c = 1, rewards 0, 1 and 0.5. Each game is one move, so each return is the reward.
    // Iterations 1 to 3 expand the actions in order: N = 1, 1, 1 and Q = 0, 1, 0.5.
    // Then the root is fully expanded, and UCT picks the highest Q + sqrt(ln N(root) / N(child)):
    //   4. ln 3 = 1.0986: 0 + 1.0481, 1 + 1.0481, 0.5 + 1.0481           -> action 1, N = 1, 2, 1
    //   5. ln 4 = 1.3863: 0 + 1.1774, 1 + sqrt(1.3863 / 2) = 1.8326, 1.6774  -> action 1, N = 1, 3, 1
    //   6. ln 5 = 1.6094: 0 + 1.2686, 1 + sqrt(1.6094 / 3) = 1.7325, 0.5 + 1.2686 = 1.7686
    //                                                                    -> action 2, N = 1, 3, 2
    relearn::mcts<one_move> search(one_move(), 6, 1.0);
    search.search(0);
    const auto & tree = search.tree();
    REQUIRE(tree.size() == 4);
    REQUIRE(tree[0].visits == 6);
    REQUIRE(tree[0].children.size() == 3);
    const auto & a0 = tree[tree[0].children[0]];
    const auto & a1 = tree[tree[0].children[1]];
    const auto & a2 = tree[tree[0].children[2]];
    REQUIRE(a0.visits == 1);
    REQUIRE(a1.visits == 3);
    REQUIRE(a2.visits == 2);
    REQUIRE(a0.value() == Approx(0));
    REQUIRE(a1.value() == Approx(1));
    REQUIRE(a2.value() == Approx(0.5));
    REQUIRE(search.best_action() == 1);
    // searching again starts a new tree
    REQUIRE(search(0) == 1);
    REQUIRE(search.tree().size() == 4);
    REQUIRE(search.tree()[0].visits == 6);
    // after 3 iterations every action has 1 visit: the tie goes to the highest value, action 1
    search.iterations = 3;
    REQUIRE(search(0) == 1);
    // and with 2, actions 0 and 1 are tied: action 1 again
    search.iterations = 2;
    REQUIRE(search(0) == 1);
}

/// verifies that returns are discounted, both in the tree and in the rollout
TEST_CASE("mcts: returns are discounted", "[mcts]")
{
    // γ = 0.5, the chain earns 0, 0, then 1. Each iteration adds one node:
    //   1. node 1 (state 1), rollout 1 -> 2 -> 3: G = 0 + 0.5 * 1 = 0.5 from state 1, so node 1: 0 + 0.5 * 0.5 = 0.25
    //   2. node 2 (state 2), rollout 2 -> 3: G = 1, so node 2: 0 + 0.5 * 1 = 0.5 and node 1: 0 + 0.5 * 0.5 = 0.25
    //   3. node 3 (the end), no rollout: node 3: 1, node 2: 0 + 0.5 * 1 = 0.5, node 1: 0.25
    relearn::mcts<chain> search(chain(), 1);
    search.gamma = 0.5;
    search.search(0);
    REQUIRE(search.tree().size() == 2);
    REQUIRE(search.tree()[1].visits == 1);
    REQUIRE(search.tree()[1].value() == Approx(0.25));

    search.iterations = 3;
    search.search(0);
    const auto & tree = search.tree();
    REQUIRE(tree.size() == 4);
    REQUIRE(tree[1].visits == 3);
    REQUIRE(tree[2].visits == 2);
    REQUIRE(tree[3].visits == 1);
    REQUIRE(tree[1].value() == Approx(0.25));
    REQUIRE(tree[2].value() == Approx(0.5));
    REQUIRE(tree[3].value() == Approx(1));
    REQUIRE(tree[3].terminal);
    REQUIRE(tree[3].actions.empty());
}

/// verifies that a rollout cut short counts only the rewards it received
TEST_CASE("mcts: a rollout cut short", "[mcts]")
{
    // γ = 1: the rollout from state 1 needs 2 steps to reach the reward of 1
    relearn::mcts<chain> search(chain(), 1);
    search.gamma = 1;
    search.rollout_limit = 2;
    search.search(0);
    REQUIRE(search.tree()[1].value() == Approx(1));
    // one step is not enough: it only receives 0
    search.rollout_limit = 1;
    search.search(0);
    REQUIRE(search.tree()[1].value() == Approx(0));
    // no rollout at all: the new node is worth 0, until the tree itself reaches the end
    search.rollout_limit = 0;
    search.search(0);
    REQUIRE(search.tree()[1].value() == Approx(0));
}

/// verifies that each value is from the point of view of the player who moved (negamax)
TEST_CASE("mcts: two players, each value is the mover's", "[mcts][two_players]")
{
    // c = 1. Left earns player 0 0.5, then player 1 0.25; right earns 0, then player 1 1.
    //   1. expand left, rollout: player 1 earns 0.25. Player 1 moved next, so the sign changes:
    //      left: G = 0.5 - 0.25 = 0.25
    //   2. expand right, rollout: player 1 earns 1, so right: G = 0 - 1 = -1
    //   3. ln 2 = 0.6931: left 0.25 + 0.8326, right -1 + 0.8326 -> left, and expand its first move,
    //      which ends the game: that node earns player 1 0.25 (G = 0.25), and left: G = 0.5 - 0.25 = 0.25
    //   4. ln 3 = 1.0986: left 0.25 + sqrt(1.0986 / 2) = 0.9912, right -1 + 1.0481 = 0.0481
    //      -> left, and expand its second move: the same again
    // Without the change of sign, left would be worth 0.75 and right 1, and player 0 would go right.
    relearn::mcts<two_player_tree> search(two_player_tree(), 4, 1.0);
    search.search(0);
    const auto & tree = search.tree();
    REQUIRE(tree[0].visits == 4);
    const auto & left  = tree[tree[0].children[0]];
    const auto & right = tree[tree[0].children[1]];
    REQUIRE(left.player == 1);
    REQUIRE(left.visits == 3);
    REQUIRE(left.value() == Approx(0.25));
    REQUIRE(right.visits == 1);
    REQUIRE(right.value() == Approx(-1));
    // player 1's moves after left are worth 0.25 to player 1
    REQUIRE(left.children.size() == 2);
    for (auto c : left.children) {
        REQUIRE(tree[c].visits == 1);
        REQUIRE(tree[c].value() == Approx(0.25));
    }
    REQUIRE(search.best_action() == 0);

    // with many iterations, every node goes to its true value, from its mover's point of view
    search.iterations = 2000;
    search.search(0);
    REQUIRE(search.tree()[search.tree()[0].children[0]].value() == Approx(0.25));
    REQUIRE(search.tree()[search.tree()[0].children[1]].value() == Approx(-1));
    REQUIRE(search.best_action() == 0);
}

/// verifies the sign of the return when the player changes during the rollout
TEST_CASE("mcts: two players, the rollout changes sign too", "[mcts][two_players]")
{
    // γ = 0.5. Player 0 moves (0), player 1 moves (0), and player 0's last move earns player 0 1.
    //   1. node 1 (player 1 to move), rollout: player 1's move, then player 0's, which earns 1.
    //      Backwards: G = 1 for player 0, then player 1's move: G = 0 - 0.5 * 1 = -0.5 for player 1,
    //      then node 1, made by player 0: G = 0 - 0.5 * (-0.5) = 0.25
    //   2. node 2 (player 0 to move), rollout: player 0's move earns 1, G = 1;
    //      node 2, made by player 1: G = 0 - 0.5 * 1 = -0.5; node 1: 0 - 0.5 * (-0.5) = 0.25
    //   3. node 3 (the end), made by player 0: G = 1; node 2: -0.5; node 1: 0.25
    relearn::mcts<two_player_chain> search(two_player_chain(), 1);
    search.gamma = 0.5;
    search.search(0);
    REQUIRE(search.tree()[1].value() == Approx(0.25));
    search.iterations = 3;
    search.search(0);
    const auto & tree = search.tree();
    REQUIRE(tree.size() == 4);
    REQUIRE(tree[1].value() == Approx(0.25));
    REQUIRE(tree[2].value() == Approx(-0.5));
    REQUIRE(tree[3].value() == Approx(1));
}

/// verifies that the move played is the most visited one, even when another has a higher value
TEST_CASE("mcts: the move played is the most visited", "[mcts]")
{
    // c = 1, no rollouts (so the leaves are worth 0 until the tree reaches the end).
    // Safe ends with 0.6; deep earns 0, then 1 one move later.
    //   1. expand safe: 0.6          2. expand deep: 0 (no rollout)
    //   3. ln 2: safe 0.6 + 0.8326, deep 0 + 0.8326                 -> safe, N = 2, 1
    //   4. ln 3: safe 0.6 + 0.7412, deep 0 + 1.0481                 -> safe, N = 3, 1
    //   5. ln 4: safe 0.6 + sqrt(1.3863 / 3) = 1.2798, deep 1.1774  -> safe, N = 4, 1
    //   6. ln 5: safe 0.6 + sqrt(1.6094 / 4) = 1.2343, deep 1.2686  -> deep, which expands its move
    //      and reaches the end: G = 0 + 1 = 1, deep N = 2, Q = (0 + 1) / 2 = 0.5
    //   7. ln 6: safe 0.6 + sqrt(1.7918 / 4) = 1.2693, deep 0.5 + sqrt(1.7918 / 2) = 1.4465
    //      -> deep, G = 1, N = 3, Q = 2/3
    // Deep has the higher value, 2/3 against 0.6, but safe has more visits, 4 against 3: safe is played
    relearn::mcts<safe_or_deep> search(safe_or_deep(), 7, 1.0);
    search.rollout_limit = 0;
    search.search(0);
    const auto & tree = search.tree();
    const auto & safe = tree[tree[0].children[0]];
    const auto & deep = tree[tree[0].children[1]];
    REQUIRE(safe.visits == 4);
    REQUIRE(safe.value() == Approx(0.6));
    REQUIRE(deep.visits == 3);
    REQUIRE(deep.value() == Approx(2.0 / 3.0));
    REQUIRE(search.best_action() == 0);
    // one more iteration goes to deep: N = 4 and 4, Q = 3/4 against 0.6, and the tie goes to the higher value
    search.iterations = 8;
    search.search(0);
    REQUIRE(search.tree()[search.tree()[0].children[1]].visits == 4);
    REQUIRE(search.best_action() == 1);
    // and another one: deep then has the most visits, 5 against 4
    search.iterations = 9;
    search.search(0);
    REQUIRE(search.tree()[search.tree()[0].children[1]].visits == 5);
    REQUIRE(search.best_action() == 1);
}

/// verifies the constructor, and brace initialisation
/**
 * @struct long_names
 * @brief a one-player chain whose states are long strings (kept on the heap): each step adds a letter,
 *        and the third step ends the game with a reward of 1
 */
struct long_names
{
    using state_type  = std::string;
    using action_type = int;

    /// a step's outcome, with a string for the next state
    struct result
    {
        std::string next;
        double reward;
        bool terminated;
    };

    std::vector<int> actions(const std::string &) const
    {
        return {0, 1};
    }

    result step(const std::string & s,
                const int & a) const
    {
        const bool ends = s.size() >= 42;
        return {s + (a == 0 ? "a" : "b"), ends ? 1.0 : 0.0, ends};
    }

    unsigned int player(const std::string &) const
    {
        return 0;
    }
};

/// verifies that a search can start from a state of the tree itself, which the new tree replaces
TEST_CASE("mcts: a search from a state of the last tree", "[mcts][regression]")
{
    // 40 letters: far too long for the string to fit in place, so it lives on the heap
    const std::string start(40, 's');
    relearn::mcts<long_names> planner(long_names(), 50);
    planner.search(start);
    REQUIRE(planner.tree().size() > 1);
    // the first child's state belongs to the tree that the next search throws away
    const std::string child = planner.tree()[planner.tree()[0].children[0]].state;
    planner.search(planner.tree()[planner.tree()[0].children[0]].state);
    REQUIRE(planner.tree()[0].state == child);
    REQUIRE(planner.tree()[0].visits == 50);
}

TEST_CASE("mcts: constructor and brace initialisation", "[mcts]")
{
    relearn::mcts<one_move> defaults(one_move{});
    REQUIRE(defaults.iterations == 1000);
    REQUIRE(defaults.exploration == Approx(std::sqrt(2.0)));
    REQUIRE(defaults.gamma == Approx(1));
    REQUIRE(defaults.rollout_limit == 1000);
    auto braced = relearn::mcts<one_move>{one_move{}, 50, 0.5, 7};
    REQUIRE(braced.iterations == 50);
    REQUIRE(braced.exploration == Approx(0.5));
    // float values
    relearn::mcts<one_move,float> small(one_move{}, 6, 1.0f);
    REQUIRE(small(0) == 1);
    REQUIRE(small.tree()[small.tree()[0].children[2]].value() == Approx(0.5f));
}

/// verifies that it takes a win, and blocks the opponent's win, in a few positions
TEST_CASE("mcts: tic-tac-toe, it wins at once and blocks", "[mcts][tic_tac_toe]")
{
    //  X | X | .        X | . | O        X | X | .        X | X | .
    //  O | O | .        . | O | .        . | O | .        O | O | .
    //  . | . | .        . | . | X        . | . | .        . | . | X
    //  X wins at 2      X blocks at 6    O blocks at 2    O wins at 5
    //  (not block 5)    (then forks)                      (not block 2)
    // Over 10 seeds, with libstdc++ and with libc++, every seed found every move with 100 iterations;
    // with 70, one move was missed (libc++), and with 50, four (libstdc++). The test uses 200
    struct puzzle
    {
        games::board b;
        unsigned int answer;
    };
    std::vector<puzzle> puzzles = {{position({0, 1}, {3, 4}), 2},
                                   {position({0, 8}, {2, 4}), 6},
                                   {position({0, 1}, {4}), 2},
                                   {position({0, 1, 8}, {3, 4}), 5}};
    for (unsigned int seed = 1; seed <= 10; seed++) {
        planner p(games::tic_tac_toe_model(), 200, exploration, seed);
        for (const auto & q : puzzles) {
            REQUIRE(p(q.b) == q.answer);
        }
    }
}

/// verifies that it never loses to a perfect opponent, as X and as O
TEST_CASE("mcts: tic-tac-toe, it never loses to a perfect opponent", "[mcts][tic_tac_toe]")
{
    // The perfect opponent picks at random between its best moves, so that the games are not all the same.
    // Over 10 seeds, 300 games as X and 300 as O for each, MCTS lost (libstdc++ and libc++) 260 and 254
    // of the 6000 games with 800 iterations, 37 and 34 with 1600, 6 and 5 with 2400, and none with
    // 3200, 6400 or 10000. The test uses 10000, about three times the first count without a loss
    games::tic_tac_toe solver;
    std::mt19937 prng(1);
    planner p(games::tic_tac_toe_model(), 10000, exploration, 1);
    auto perfect = perfect_player(solver, prng);
    auto agent = mcts_player(p);
    for (int i = 0; i < 25; i++) {
        REQUIRE(match(agent, perfect) != 2);
        REQUIRE(match(perfect, agent) != 1);
    }
    // the perfect opponent of games::tic_tac_toe itself, which answers X inside step()
    games::tic_tac_toe game(games::tic_tac_toe::perfect);
    for (int i = 0; i < 5; i++) {
        auto r = games::play(game, [&](const games::board & b, const std::vector<unsigned int> &) {
            return p(b);
        }, 9);
        REQUIRE(r.terminated);
        REQUIRE(r.total_reward == 0);
    }
}

/// verifies that it almost always beats a random opponent
TEST_CASE("mcts: tic-tac-toe, it beats a random opponent", "[mcts][tic_tac_toe]")
{
    // 5000 iterations, 100 games as X and 100 as O. Over 10 seeds, with libstdc++ and with libc++,
    // it never lost, and won 97 to 100 games as X and 87 to 95 as O (as O it moves second, and a
    // random X sometimes stumbles into a draw). The sweep of the test above also played 600 games against
    // the random opponent per seed: it still lost one with 1600 iterations, and none from 2400 on
    std::mt19937 prng(2);
    planner p(games::tic_tac_toe_model(), 5000, exploration, 2);
    auto agent = mcts_player(p);
    auto opponent = random_player(prng);
    int x_wins = 0;
    int o_wins = 0;
    for (int i = 0; i < 100; i++) {
        auto as_x = match(agent, opponent);
        auto as_o = match(opponent, agent);
        REQUIRE(as_x != 2);
        REQUIRE(as_o != 1);
        x_wins += as_x == 1;
        o_wins += as_o == 2;
    }
    REQUIRE(x_wins >= 90);
    REQUIRE(o_wins >= 80);
}
