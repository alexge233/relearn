// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/dyna_maze.hpp"
#include <cmath>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of `dyna_q` (Sutton & Barto, Sections 8.2 and 8.3):
 * - one update at a time, against values worked out by hand
 * - the model: what it remembers, and the imagined (planning) steps it gives
 * - Dyna-Q+: the exploration bonus, and the actions it has not tried yet
 * - the Dyna maze (Figure 8.2), the blocking maze (Figure 8.4) and the shortcut maze (Figure 8.5)
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;
using learner_type = relearn::dyna_q<state,action>;

using game_type    = games::dyna_maze;
using types        = games::types<game_type>;
using maze_learner = relearn::dyna_q<types::state, types::action>;

/**
 * @brief a corridor walked from left to right: s0, s1, s2, s3, and the last move earns 1
 * @return the four links of the episode, which ends after the last one
 */
episode corridor()
{
    return episode {{state(0, "s0"), action("right")},
                    {state(0, "s1"), action("right")},
                    {state(0, "s2"), action("right")},
                    {state(1, "s3"), action("right")}};
}

/**
 * @brief let a Dyna-Q+ learner know every move of the maze
 * @param learner is the learner
 * @param maze is the maze it learns
 */
void know_the_moves(maze_learner & learner,
                    const game_type & maze)
{
    learner.legal_actions = [&maze](const types::state & s) {
        std::vector<types::action> moves;
        for (const auto & a : maze.actions(s.trait())) {
            moves.push_back(types::action(a));
        }
        return moves;
    };
}

/**
 * @brief the mean number of moves of episodes 2 to 10 in the Dyna maze (Figure 8.2)
 * @param planning_steps is n
 * @param runs is the number of runs, each a new agent learning from scratch
 * @return the mean over the runs
 *
 * The settings of Sutton & Barto, Figure 8.2: ε-greedy with ε = 0.1, α = 0.1, γ = 0.95.
 */
double early_moves(unsigned int planning_steps,
                   unsigned int runs)
{
    double moves = 0;
    for (unsigned int run = 1; run <= runs; run++) {
        std::mt19937 prng(run);
        game_type maze;
        types::policy memory;
        maze_learner learner(0.1, 0.95, planning_steps, run + 1000);
        // the ε-greedy agent
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        for (unsigned int i = 1; i <= 10; i++) {
            auto r = games::play_online(maze, agent, learner, memory, 100000);
            // the first episode is a random walk for every n: leave it out
            if (i >= 2) {
                moves += r.steps / (9.0 * runs);
            }
        }
    }
    return moves;
}

/// what one run in a changing maze achieved
struct changing_run
{
    /// the goals reached in the whole run
    double goals = 0;
    /// the moves per goal in the last 1000 moves of the run
    double moves_per_goal = 0;
};

/**
 * @brief learn a maze that changes, with Dyna-Q or Dyna-Q+
 * @param layout is the blocking or the shortcut maze
 * @param change is the move at which the maze changes
 * @param total is the number of moves of the run
 * @param kappa is κ: 0 for Dyna-Q
 * @param seed seeds the run
 * @return the goals reached, and the moves per goal at the end of the run
 *
 * ε-greedy with ε = 0.1, α = 1 (the maze is deterministic), γ = 0.95, n = 50.
 */
changing_run learn_changing(game_type::layout_type layout,
                            unsigned long change,
                            unsigned long total,
                            double kappa,
                            unsigned int seed)
{
    std::mt19937 prng(seed);
    game_type maze(layout, change);
    types::policy memory;
    maze_learner learner(1.0, 0.95, 50, seed + 1000);
    learner.kappa = kappa;
    know_the_moves(learner, maze);
    // the ε-greedy agent
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
    };
    changing_run result;
    double late_goals = 0;
    while (maze.moves() < total) {
        const auto before = maze.moves();
        auto r = games::play_online(maze, agent, learner, memory, total - maze.moves());
        result.goals += r.total_reward;
        // the episodes that start in the last 1000 moves
        if (before >= total - 1000) {
            late_goals += r.total_reward;
        }
    }
    result.moves_per_goal = late_goals > 0 ? 1000 / late_goals : 1000;
    return result;
}
} // namespace

/// verifies one real step, and then the imagined steps the model gives
TEST_CASE("dyna_q: one real step, then n imagined steps", "[dyna_q]")
{
    // α = 0.5, n = 3, and one step that ends the episode with r = 1:
    // - the real step: Q = 0 + 0.5 * (1 - 0) = 0.5
    // - the model knows only this step, so all three imagined steps are this one again:
    //   0.75, 0.875, 0.9375
    learner_type learner(0.5, 0.9, 3, 1);
    policy memory;
    learner(episode {{state(1, "s0"), action("a")}}, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(0.9375));
    REQUIRE(learner.steps() == 1);
    REQUIRE(learner.remembers(state("s0"), action("a")));
}

/// verifies that without planning, Dyna-Q is Q-learning
TEST_CASE("dyna_q: no planning is Q-learning", "[dyna_q]")
{
    // the same episodes, played at random in the Dyna maze, learnt by both
    std::mt19937 prng(7);
    game_type maze;
    types::policy random_memory;
    auto random_agent = [&](const game_type::state_type & s,
                            const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(random_memory, s, legal, 1.0, prng);
    };
    types::policy q_memory;
    types::policy dyna_memory;
    relearn::q_learning<types::state, types::action> q(0.5, 0.95);
    maze_learner dyna(0.5, 0.95, 0, 1);
    for (unsigned int i = 0; i < 20; i++) {
        auto r = games::play(maze, random_agent, 300);
        q(r.episode, q_memory, r.terminated);
        dyna(r.episode, dyna_memory, r.terminated);
    }
    // every value is exactly the same
    for (int x = 0; x < game_type::width; x++) {
        for (int y = 0; y < game_type::height; y++) {
            for (unsigned int a : {games::up, games::right, games::down, games::left}) {
                types::state s(games::position{x, y});
                REQUIRE(dyna_memory.value(s, types::action(a)) == q_memory.value(s, types::action(a)));
            }
        }
    }
}

/// verifies that planning carries a reward back along a path walked once
TEST_CASE("dyna_q: planning carries the reward back along the path", "[dyna_q]")
{
    // α = 1, γ = 0.9, one walk along the corridor. Q-learning learns only the last move:
    // the earlier ones bootstrap from states not yet valued. With n = 200 imagined steps
    // after each real one, the reward reaches the start: Q(s0) = 0.9^3, Q(s1) = 0.9^2, ...
    policy q_memory;
    relearn::q_learning<state,action> q(1.0, 0.9);
    q(corridor(), q_memory);
    REQUIRE(q_memory.value(state("s0"), action("right")) == Approx(0));
    REQUIRE(q_memory.value(state("s3"), action("right")) == Approx(1));

    policy dyna_memory;
    learner_type dyna(1.0, 0.9, 200, 1);
    dyna(corridor(), dyna_memory);
    REQUIRE(dyna_memory.value(state("s0"), action("right")) == Approx(0.729));
    REQUIRE(dyna_memory.value(state("s1"), action("right")) == Approx(0.81));
    REQUIRE(dyna_memory.value(state("s2"), action("right")) == Approx(0.9));
    REQUIRE(dyna_memory.value(state("s3"), action("right")) == Approx(1));
}

/// verifies that the model keeps the last outcome of each pair, so it notices a change
TEST_CASE("dyna_q: the model keeps the latest outcome of a move", "[dyna_q]")
{
    // α = 1, n = 10. First a ends the game from s0 with r = 1; then the world changes,
    // and a leads to s1 with r = 0. s1 has no value yet, so every imagined step of
    // (s0, a) now gives 0 + 0.9 * 0 = 0, not 1.
    learner_type learner(1.0, 0.9, 10, 1);
    policy memory;
    learner(episode {{state(1, "s0"), action("a")}}, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(1));
    learner(episode {{state(0, "s0"), action("a")},
                     {state(0, "s1"), action("b")}}, memory, false);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(0));
}

/// verifies that a cut-short episode's last link is neither learnt nor remembered
TEST_CASE("dyna_q: the last link of a cut-short episode is not learnt", "[dyna_q]")
{
    learner_type learner(1.0, 0.9, 5, 1);
    policy memory;
    auto walk = corridor();
    learner(walk, memory, false);
    REQUIRE(learner.steps() == 3);
    REQUIRE(learner.remembers(state("s2"), action("right")));
    REQUIRE_FALSE(learner.remembers(state("s3"), action("right")));
    REQUIRE(memory.value(state("s3"), action("right")) == Approx(0));
}

/// verifies Dyna-Q+'s bonus: κ * sqrt(τ), τ being the real steps since a pair was last tried
TEST_CASE("dyna_q+: the bonus grows with the steps since a move was tried", "[dyna_q][dyna_q_plus]")
{
    // α = 1, κ = 0.1, no planning whilst learning. (s0, a) is tried at step 1, then
    // (s1, b) at steps 2, 3 and 4; both end the game with r = 0. Planning afterwards:
    // - (s0, a) was last tried 3 steps ago: Q = 0 + 0.1 * sqrt(3)
    // - (s1, b) has just been tried:        Q = 0 + 0.1 * sqrt(0) = 0
    learner_type learner(1.0, 0.9, 0, 1);
    learner.kappa = 0.1;
    policy memory;
    learner(episode {{state(0, "s0"), action("a")}}, memory);
    for (unsigned int i = 0; i < 3; i++) {
        learner(episode {{state(0, "s1"), action("b")}}, memory);
    }
    REQUIRE(learner.steps() == 4);
    learner.planning_steps = 50;
    learner.plan(memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(0.1 * std::sqrt(3.0)));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(0));
}

/// verifies that Dyna-Q+ plans with the actions it has not tried, as moves back to the same state
TEST_CASE("dyna_q+: actions not tried yet are imagined to lead back with a reward of 0", "[dyna_q][dyna_q_plus]")
{
    // s0 has two actions, a and b. a is tried once, and ends the game with r = 0.
    // α = 1, γ = 0.5, κ = 0.1. b has never been tried: it is imagined to lead back to s0,
    // with a reward of 0, as if last tried at step 0, so τ = 1 and its bonus is 0.1.
    // Planning repeats Q(s0, b) = 0.1 + 0.5 * max(Q(s0, a), Q(s0, b)), with Q(s0, a) = 0,
    // which settles at 0.1 / (1 - 0.5) = 0.2
    auto both = [](const state &) {
        return std::vector<action> {action("a"), action("b")};
    };
    learner_type plus(1.0, 0.5, 200, 1);
    plus.kappa = 0.1;
    plus.legal_actions = both;
    policy memory;
    plus(episode {{state(0, "s0"), action("a")}}, memory);
    REQUIRE(memory.value(state("s0"), action("b")) == Approx(0.2));
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(0));
    // b is only imagined: it has not really been tried
    REQUIRE(plus.remembers(state("s0"), action("a")));
    REQUIRE_FALSE(plus.remembers(state("s0"), action("b")));

    // Dyna-Q (κ = 0) never imagines actions it has not tried, even if it knows them
    learner_type dyna(1.0, 0.5, 200, 1);
    dyna.legal_actions = both;
    policy dyna_memory;
    dyna(episode {{state(0, "s0"), action("a")}}, dyna_memory);
    REQUIRE(dyna_memory.actions(state("s0")).size() == 1);
}

/// verifies the constructor, and brace initialisation
TEST_CASE("dyna_q: constructor and brace initialisation", "[dyna_q]")
{
    learner_type defaults;
    REQUIRE(defaults.alpha == Approx(0.9));
    REQUIRE(defaults.gamma == Approx(0.9));
    REQUIRE(defaults.planning_steps == 5);
    REQUIRE(defaults.kappa == Approx(0));
    REQUIRE(defaults.steps() == 0);
    auto learner = learner_type{0.1, 0.95, 50, 7};
    REQUIRE(learner.alpha == Approx(0.1));
    REQUIRE(learner.gamma == Approx(0.95));
    REQUIRE(learner.planning_steps == 50);
    // the seed is the generator's: the same seed draws the same numbers
    std::mt19937 same(7);
    REQUIRE(learner.prng() == same());
}

/// verifies Figure 8.2: the more it plans, the fewer episodes Dyna-Q needs to learn the maze
TEST_CASE("dyna_q: Figure 8.2, more planning learns the maze in fewer episodes", "[dyna_q][dyna_maze]")
{
    // over 10 groups of 10 runs, the mean moves of episodes 2 to 10 were
    // 265 to 363 (n = 0), 37 to 44 (n = 5) and 19 to 22 (n = 50)
    const double none = early_moves(0, 10);
    const double some = early_moves(5, 10);
    const double many = early_moves(50, 10);
    REQUIRE(none > 3 * some);
    REQUIRE(some > 1.5 * many);
    // with ε = 0.1, the shortest path (14 moves) takes about 17 moves on average
    REQUIRE(many < 25);
}

/// verifies Figure 8.4: when the short way is blocked, Dyna-Q+ finds the new way, every time
TEST_CASE("dyna_q+: Figure 8.4, the blocking maze", "[dyna_q][dyna_q_plus][dyna_maze]")
{
    // the wall moves after 1000 moves, and the runs last 3000. Over 10 groups of 10 runs,
    // Dyna-Q+ found the new way in all runs (at most 25 moves per goal at the end), and
    // Dyna-Q in 0 to 5; Dyna-Q+ reached 151 to 161 goals per run, Dyna-Q 47 to 92
    double goals = 0;
    double plus_goals = 0;
    for (unsigned int seed = 1; seed <= 10; seed++) {
        auto dyna = learn_changing(game_type::blocking, 1000, 3000, 0, seed);
        auto plus = learn_changing(game_type::blocking, 1000, 3000, 0.001, seed);
        REQUIRE(plus.moves_per_goal <= 25);
        goals += dyna.goals;
        plus_goals += plus.goals;
    }
    REQUIRE(plus_goals > 1.5 * goals);
}

/// verifies Figure 8.5: when a shortcut opens, Dyna-Q+ finds it, and Dyna-Q never does
TEST_CASE("dyna_q+: Figure 8.5, the shortcut maze", "[dyna_q][dyna_q_plus][dyna_maze]")
{
    // the shortcut opens after 3000 moves, and the runs last 6000. Taking it costs 10 moves,
    // the old way 16, so at most 13 moves per goal at the end means the shortcut is taken.
    // Over 10 groups of 10 runs, Dyna-Q+ took it in 6 to 9 runs, and Dyna-Q in none;
    // Dyna-Q+ reached 350 to 359 goals per run, Dyna-Q 281 to 302
    unsigned int taken = 0;
    unsigned int plus_taken = 0;
    double goals = 0;
    double plus_goals = 0;
    for (unsigned int seed = 1; seed <= 10; seed++) {
        auto dyna = learn_changing(game_type::shortcut, 3000, 6000, 0, seed);
        auto plus = learn_changing(game_type::shortcut, 3000, 6000, 0.001, seed);
        taken += dyna.moves_per_goal <= 13;
        plus_taken += plus.moves_per_goal <= 13;
        goals += dyna.goals;
        plus_goals += plus.goals;
    }
    REQUIRE(taken == 0);
    REQUIRE(plus_taken >= 5);
    REQUIRE(plus_goals > 1.1 * goals);
}
