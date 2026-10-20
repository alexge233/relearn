// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/dyna_maze.hpp"
#include "solvers.hpp"
#include <cmath>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of `prioritized_sweeping` (Sutton & Barto, Section 8.4):
 * - the queue, its priorities and the threshold θ, against values worked out by hand
 * - the model and the predecessors of each state
 * - the Dyna maze learnt from random play: the exact values (from `solvers.hpp`) and the shortest path
 * - Example 8.4: in mazes of increasing size, far fewer updates than Dyna-Q to find a good path
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;
using learner_type = relearn::prioritized_sweeping<state,action>;

using game_type    = games::dyna_maze;
using types        = games::types<game_type>;
using maze_learner = relearn::prioritized_sweeping<types::state, types::action>;
using dyna_learner = relearn::dyna_q<types::state, types::action>;

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
 * @brief the exact model of a Dyna maze, written from its rules, for `solvers::tabular`
 * @param maze is the maze (its walls, its size, and its goal)
 * @param gamma is the discount
 * @return the model: state y * columns + x, actions up, right, down, left
 */
solvers::tabular maze_model(const game_type & maze,
                            double gamma)
{
    solvers::tabular m;
    m.states  = maze.columns() * maze.rows();
    m.actions = 4;
    m.gamma   = gamma;
    m.model = [maze](int s, int a) {
        std::vector<solvers::outcome> out;
        games::position from = {s % maze.columns(), s / maze.columns()};
        // the goal and the walls are not states the agent can be in
        if (from == maze.goal() || maze.wall(from)) {
            return out;
        }
        // a move into a wall, or off the grid, stays put; reaching the goal is worth 1 and ends it
        auto to = games::moved(from, static_cast<unsigned int>(a), maze.columns(), maze.rows());
        if (maze.wall(to)) {
            to = from;
        }
        const bool goal = to == maze.goal();
        out.push_back({1.0, to.y * maze.columns() + to.x, goal ? 1.0 : 0.0, goal});
        return out;
    };
    return m;
}

/**
 * @brief the length of the greedy path from the start to the goal
 * @param maze is the maze
 * @param memory is the learnt policy
 * @param prng breaks ties
 * @param limit is the longest path tried
 * @return the number of moves, or limit + 1 if the goal is not reached in `limit` moves
 */
std::size_t greedy_path(game_type & maze,
                        const types::policy & memory,
                        std::mt19937 & prng,
                        std::size_t limit)
{
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::greedy<game_type>(memory, s, legal, prng);
    };
    auto r = games::play(maze, agent, limit);
    return r.terminated ? r.steps : limit + 1;
}

/**
 * @brief the updates a learner needs, playing ε-greedy, before its greedy path is a good one (Example 8.4)
 * @param scale is the scale of the maze: 1 is the maze of Example 8.1, 2 is twice as fine, ...
 * @param learner is Dyna-Q or prioritized sweeping
 * @param updates gives the updates the learner has made so far
 * @param seed seeds the agent
 * @return the updates made when, after an episode, the greedy path is at most 20% longer than the shortest
 *
 * ε-greedy with ε = 0.1, as in Figure 8.2; the learners are set up by the caller.
 */
template <class learner_class,
          class counter>
std::size_t updates_until_good(int scale,
                               learner_class & learner,
                               counter updates,
                               unsigned int seed)
{
    std::mt19937 prng(seed);
    std::mt19937 ties(seed + 500);
    game_type maze(game_type::maze, 1000, scale);
    // the shortest path is 14 moves at scale 1, and scales with the maze
    const std::size_t good = static_cast<std::size_t>(1.2 * 14 * scale);
    types::policy memory;
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
    };
    for (unsigned int i = 0; i < 100000; i++) {
        games::play_online(maze, agent, learner, memory, 1000000);
        if (greedy_path(maze, memory, ties, good) <= good) {
            return updates(learner);
        }
    }
    return 0;
}

/// @return the updates of Dyna-Q: each real step is one, followed by n imagined ones
std::size_t dyna_updates(const dyna_learner & learner)
{
    return learner.steps() * (learner.planning_steps + 1);
}

/// @return the updates of prioritized sweeping: one for each move taken from the queue
std::size_t sweeping_updates(const maze_learner & learner)
{
    return learner.updates();
}
} // namespace

/// verifies that one walk along the corridor carries the reward back to the start, in four updates
TEST_CASE("prioritized_sweeping: the reward is swept back along the path", "[prioritized_sweeping]")
{
    // α = 1, γ = 0.9, n = 10. The first three moves change nothing (r = 0, and every value is 0),
    // so their priority is 0 and they are not queued. The last move ends the game with r = 1:
    // P = |1 - 0| = 1, so it is queued, and planning takes it at once:
    // - Q(s3) = 1; its predecessor (s2, right) has P = |0 + 0.9 * 1 - 0| = 0.9, and is queued
    // - Q(s2) = 0.9; then (s1, right), P = 0.81: Q(s1) = 0.81; then (s0, right), P = 0.729: Q(s0) = 0.729
    // s0 has no predecessor, so the queue is empty: 4 updates, where Dyna-Q needs about 200
    learner_type learner(1.0, 0.9, 10);
    policy memory;
    learner(corridor(), memory);
    REQUIRE(memory.value(state("s0"), action("right")) == Approx(0.729));
    REQUIRE(memory.value(state("s1"), action("right")) == Approx(0.81));
    REQUIRE(memory.value(state("s2"), action("right")) == Approx(0.9));
    REQUIRE(memory.value(state("s3"), action("right")) == Approx(1));
    REQUIRE(learner.updates() == 4);
    REQUIRE(learner.steps() == 4);
    REQUIRE(learner.queued() == 0);
}

/// verifies that planning stops after n updates, and that the queue waits for the next real step
TEST_CASE("prioritized_sweeping: at most n updates after each real step", "[prioritized_sweeping]")
{
    // the same walk with n = 2: Q(s3) = 1 and Q(s2) = 0.9, and then planning stops,
    // with (s1, right) still in the queue, at priority |0 + 0.9 * 0.9 - 0| = 0.81
    learner_type learner(1.0, 0.9, 2);
    policy memory;
    learner(corridor(), memory);
    REQUIRE(memory.value(state("s3"), action("right")) == Approx(1));
    REQUIRE(memory.value(state("s2"), action("right")) == Approx(0.9));
    REQUIRE(memory.value(state("s1"), action("right")) == Approx(0));
    REQUIRE(learner.updates() == 2);
    REQUIRE(learner.queued() == 1);
    REQUIRE(learner.priority(state("s1"), action("right")) == Approx(0.81));
    // planning again takes it: Q(s1) = 0.81, and queues (s0, right) at 0.729
    learner.plan(memory);
    REQUIRE(memory.value(state("s1"), action("right")) == Approx(0.81));
    REQUIRE(memory.value(state("s0"), action("right")) == Approx(0.729));
    REQUIRE(learner.updates() == 4);
}

/// verifies that a change of θ or less is not queued
TEST_CASE("prioritized_sweeping: the threshold θ", "[prioritized_sweeping]")
{
    // γ = 0.5 and θ = 0.25: after Q(s3) = 1, (s2, right) has P = 0.5 > θ and is queued, so Q(s2) = 0.5;
    // then (s1, right) has P = 0.5 * 0.5 = 0.25, which is not more than θ: it is not queued, and stays at 0
    learner_type learner(1.0, 0.5, 10, 0.25);
    policy memory;
    learner(corridor(), memory);
    REQUIRE(memory.value(state("s2"), action("right")) == Approx(0.5));
    REQUIRE(memory.value(state("s1"), action("right")) == Approx(0));
    REQUIRE(learner.updates() == 2);
    REQUIRE(learner.queued() == 0);
}

/// verifies that a value which falls is swept back too: the priority is the size of the change
TEST_CASE("prioritized_sweeping: a fall in value is swept back too", "[prioritized_sweeping]")
{
    // α = 1, γ = 0.9, n = 10. After one walk, Q(s0) ... Q(s3) are 0.729, 0.81, 0.9 and 1 (as above).
    // Then the last move costs -1: P = |-1 - 1| = 2, so Q(s3) = -1. Its predecessor (s2, right)
    // would change by 0 + 0.9 * (-1) - 0.9 = -1.8: its priority is |-1.8| = 1.8, and so on back
    learner_type learner(1.0, 0.9, 10);
    policy memory;
    learner(corridor(), memory);
    learner(episode {{state(-1, "s3"), action("right")}}, memory);
    REQUIRE(memory.value(state("s3"), action("right")) == Approx(-1));
    REQUIRE(memory.value(state("s2"), action("right")) == Approx(-0.9));
    REQUIRE(memory.value(state("s1"), action("right")) == Approx(-0.81));
    REQUIRE(memory.value(state("s0"), action("right")) == Approx(-0.729));
    REQUIRE(learner.updates() == 8);
}

/// verifies the learning rate: each move taken from the queue moves only α of the way
TEST_CASE("prioritized_sweeping: the learning rate", "[prioritized_sweeping]")
{
    // α = 0.5, γ = 0.9, n = 10, the same walk:
    // - Q(s3) = 0 + 0.5 * (1 - 0)             = 0.5,     then (s2, right) at P = 0.9 * 0.5 = 0.45
    // - Q(s2) = 0 + 0.5 * (0.9 * 0.5 - 0)     = 0.225,   then (s1, right) at P = 0.2025
    // - Q(s1) = 0 + 0.5 * (0.9 * 0.225 - 0)   = 0.10125, then (s0, right) at P = 0.091125
    // - Q(s0) = 0 + 0.5 * (0.9 * 0.10125 - 0) = 0.0455625
    // A move is queued again only when the value of the state it leads to changes, so
    // (s3, right) waits, at 0.5, for its next real step
    learner_type learner(0.5, 0.9, 10);
    policy memory;
    learner(corridor(), memory);
    REQUIRE(memory.value(state("s3"), action("right")) == Approx(0.5));
    REQUIRE(memory.value(state("s2"), action("right")) == Approx(0.225));
    REQUIRE(memory.value(state("s1"), action("right")) == Approx(0.10125));
    REQUIRE(memory.value(state("s0"), action("right")) == Approx(0.0455625));
    REQUIRE(learner.updates() == 4);
}

/// verifies that the move of highest priority is taken first
TEST_CASE("prioritized_sweeping: the highest priority first", "[prioritized_sweeping]")
{
    // α = 1, no planning whilst learning: (s0, a) ends the game with r = 1, then (s1, b) with r = 2,
    // so they are queued at priorities 1 and 2. One update takes (s1, b), the higher one
    learner_type learner(1.0, 0.9, 0);
    policy memory;
    learner(episode {{state(1, "s0"), action("a")}}, memory);
    learner(episode {{state(2, "s1"), action("b")}}, memory);
    REQUIRE(learner.queued() == 2);
    REQUIRE(learner.priority(state("s0"), action("a")) == Approx(1));
    REQUIRE(learner.priority(state("s1"), action("b")) == Approx(2));
    learner.planning_steps = 1;
    learner.plan(memory);
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(2));
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(0));
    REQUIRE(learner.queued() == 1);
    REQUIRE(learner.priority(state("s1"), action("b")) == Approx(0));
}

/// verifies that a move queued again keeps the higher priority, and is learnt from the latest outcome
TEST_CASE("prioritized_sweeping: a move is queued once, at its higher priority", "[prioritized_sweeping]")
{
    // α = 1, no planning whilst learning: (s0, a) ends the game with r = 2 (priority 2),
    // then with r = 1 (priority |1 - 0| = 1). It is in the queue once, at priority 2,
    // but the model remembers the latest outcome, so planning learns Q = 1
    learner_type learner(1.0, 0.9, 0);
    policy memory;
    learner(episode {{state(2, "s0"), action("a")}}, memory);
    learner(episode {{state(1, "s0"), action("a")}}, memory);
    REQUIRE(learner.queued() == 1);
    REQUIRE(learner.priority(state("s0"), action("a")) == Approx(2));
    learner.planning_steps = 5;
    learner.plan(memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(1));
    REQUIRE(learner.updates() == 1);
    // and a higher priority replaces a lower one: now Q = 1, so r = 3 gives |3 - 1| = 2, then r = -4 gives |-4 - 1| = 5
    learner.planning_steps = 0;
    learner(episode {{state(3, "s0"), action("a")}}, memory);
    learner(episode {{state(-4, "s0"), action("a")}}, memory);
    REQUIRE(learner.queued() == 1);
    REQUIRE(learner.priority(state("s0"), action("a")) == Approx(5));
}

/// verifies the model, and that the predecessors follow it when the world changes
TEST_CASE("prioritized_sweeping: the model and the predecessors", "[prioritized_sweeping]")
{
    learner_type learner(1.0, 0.9, 0);
    policy memory;
    // (s0, a) leads to s1: it is a predecessor of s1
    learner(episode {{state(0, "s0"), action("a")},
                     {state(0, "s1"), action("b")}}, memory, false);
    REQUIRE(learner.remembers(state("s0"), action("a")));
    REQUIRE_FALSE(learner.remembers(state("s1"), action("b")));
    REQUIRE(learner.predecessors(state("s1")) == std::vector<step> {{state("s0"), action("a")}});
    // the world changes: (s0, a) now leads to s2, so it is a predecessor of s2, and no longer of s1
    learner(episode {{state(0, "s0"), action("a")},
                     {state(0, "s2"), action("b")}}, memory, false);
    REQUIRE(learner.predecessors(state("s1")).empty());
    REQUIRE(learner.predecessors(state("s2")) == std::vector<step> {{state("s0"), action("a")}});
    // and when it ends the game, it leads to no state at all
    learner(episode {{state(0, "s0"), action("a")}}, memory);
    REQUIRE(learner.predecessors(state("s2")).empty());
    // a move made twice is a predecessor once
    learner(episode {{state(0, "s3"), action("c")}, {state(0, "s4"), action("d")}}, memory, false);
    learner(episode {{state(0, "s3"), action("c")}, {state(0, "s4"), action("d")}}, memory, false);
    REQUIRE(learner.predecessors(state("s4")).size() == 1);
    REQUIRE(learner.steps() == 5);
}

/// verifies that the queue sweeps back along every path that leads to a change, through the predecessors
TEST_CASE("prioritized_sweeping: sweeping back along two paths", "[prioritized_sweeping]")
{
    // α = 1, γ = 0.5, n = 10. First the agent walks a → c and b → c (no reward), then c ends the game
    // with r = 1. Planning: Q(c) = 1, then both predecessors of c, (a, x) and (b, y), have priority 0.5:
    // Q(a, x) = Q(b, y) = 0.5, in 3 updates
    learner_type learner(1.0, 0.5, 10);
    policy memory;
    learner(episode {{state(0, "a"), action("x")}, {state(0, "c"), action("z")}}, memory, false);
    learner(episode {{state(0, "b"), action("y")}, {state(0, "c"), action("z")}}, memory, false);
    learner(episode {{state(1, "c"), action("z")}}, memory);
    REQUIRE(memory.value(state("c"), action("z")) == Approx(1));
    REQUIRE(memory.value(state("a"), action("x")) == Approx(0.5));
    REQUIRE(memory.value(state("b"), action("y")) == Approx(0.5));
    REQUIRE(learner.updates() == 3);
}

/// verifies that a move worth exactly 0 counts in max(Q(s, a)), though it is never worth an update
TEST_CASE("prioritized_sweeping: a move worth 0 counts in the best value of its state", "[prioritized_sweeping][regression]")
{
    // α = 1, γ = 0.9, n = 5. In b, x ends the game with r = 0 and y with r = -1, and a leads from s to b
    // with r = 0, so Q(s, a) = 0 + 0.9 * max(Q(b, x), Q(b, y)) = 0.9 * max(0, -1) = 0.
    // (b, x) has nothing to learn (its error is 0), yet it must be stored, or the best value of b would be -1
    // and Q(s, a) would be -0.9
    SECTION("x seen first") {
        learner_type learner(1.0, 0.9, 5);
        policy memory;
        learner(episode {{state(0, "b"), action("x")}}, memory);
        learner(episode {{state(-1, "b"), action("y")}}, memory);
        learner(episode {{state(0, "s"), action("a")}, {state(0, "b"), action("x")}}, memory);
        REQUIRE(memory.value(state("b"), action("x")) == Approx(0));
        REQUIRE(memory.value(state("b"), action("y")) == Approx(-1));
        REQUIRE(memory.value(state("s"), action("a")) == Approx(0));
        REQUIRE(memory.best_value(state("b")) == Approx(0));
    }
    // the other way round: Q(s, a) = -0.9 is learnt whilst only y is known in b; then x, worth 0, raises
    // the best value of b, so (s, a) is queued again (priority |0 + 0.9 * 0 - (-0.9)| = 0.9) and learns 0
    SECTION("x seen last") {
        learner_type learner(1.0, 0.9, 5);
        policy memory;
        learner(episode {{state(-1, "b"), action("y")}}, memory);
        learner(episode {{state(0, "s"), action("a")}, {state(-1, "b"), action("y")}}, memory);
        REQUIRE(memory.value(state("s"), action("a")) == Approx(-0.9));
        learner(episode {{state(0, "b"), action("x")}}, memory);
        REQUIRE(memory.value(state("s"), action("a")) == Approx(0).margin(1e-12));
        REQUIRE(memory.best_value(state("b")) == Approx(0));
    }
}

/// verifies that a copy of a learner has a queue of its own, and outlives the original
TEST_CASE("prioritized_sweeping: a copy of the learner keeps its own queue", "[prioritized_sweeping][regression]")
{
    // no planning whilst learning, so the corridor's last move stays queued (priority 1)
    learner_type * original = new learner_type(1.0, 0.9, 0);
    policy memory;
    (*original)(corridor(), memory);
    REQUIRE(original->queued() == 1);
    learner_type copy = *original;
    // the original is gone: the copy must not depend on anything of it
    delete original;
    REQUIRE(copy.queued() == 1);
    REQUIRE(copy.priority(state("s3"), action("right")) == Approx(1));
    // raising the priority of the queued move replaces its entry, in the copy's own queue
    copy(episode {{state(3, "s3"), action("right")}}, memory);
    REQUIRE(copy.queued() == 1);
    REQUIRE(copy.priority(state("s3"), action("right")) == Approx(3));
    // and planning sweeps the reward back along the corridor, as in the first test
    copy.planning_steps = 10;
    copy.plan(memory);
    REQUIRE(copy.queued() == 0);
    REQUIRE(memory.value(state("s3"), action("right")) == Approx(3));
    REQUIRE(memory.value(state("s0"), action("right")) == Approx(3 * 0.9 * 0.9 * 0.9));
}

/// verifies that a cut-short episode's last link is neither learnt nor remembered
TEST_CASE("prioritized_sweeping: the last link of a cut-short episode is not learnt", "[prioritized_sweeping]")
{
    learner_type learner(1.0, 0.9, 5);
    policy memory;
    learner(corridor(), memory, false);
    REQUIRE(learner.steps() == 3);
    REQUIRE(learner.remembers(state("s2"), action("right")));
    REQUIRE_FALSE(learner.remembers(state("s3"), action("right")));
    REQUIRE(learner.updates() == 0);
}

/// verifies that the learner works with `float` values, though a state's reward is a `double`
TEST_CASE("prioritized_sweeping: float values", "[prioritized_sweeping][regression]")
{
    // the same as the first test, in float
    using float_learner = relearn::prioritized_sweeping<state,action,episode,float>;
    float_learner learner(1.0f, 0.9f, 10, 0.0001f);
    relearn::policy<state,action,float> memory;
    learner(corridor(), memory);
    REQUIRE(memory.value(state("s0"), action("right")) == Approx(0.729f));
}

/// verifies the constructor, and brace initialisation
TEST_CASE("prioritized_sweeping: constructor and brace initialisation", "[prioritized_sweeping]")
{
    learner_type defaults;
    REQUIRE(defaults.alpha == Approx(0.9));
    REQUIRE(defaults.gamma == Approx(0.9));
    REQUIRE(defaults.planning_steps == 5);
    REQUIRE(defaults.theta == Approx(0.0001));
    REQUIRE(defaults.steps() == 0);
    REQUIRE(defaults.updates() == 0);
    REQUIRE(defaults.queued() == 0);
    auto learner = learner_type{0.5, 0.95, 50, 0.01};
    REQUIRE(learner.alpha == Approx(0.5));
    REQUIRE(learner.gamma == Approx(0.95));
    REQUIRE(learner.planning_steps == 50);
    REQUIRE(learner.theta == Approx(0.01));
}

/// verifies, from random play in the Dyna maze, the exact values and the shortest path
TEST_CASE("prioritized_sweeping: the Dyna maze from random play, exactly", "[prioritized_sweeping][dyna_maze]")
{
    // a random agent tries every move of the maze (at scales 1 and 2) many times. With α = 1
    // (the maze is deterministic) and θ = 1e-9, the values are then the optimal ones,
    // Q(s,a) = r + γ * V(s'), with V worked out by value iteration in `solvers.hpp`,
    // and the greedy path is a shortest one: 14 moves, and 28 at scale 2.
    // Over 10 groups of 10 seeds, with libstdc++ and libc++, 50 random episodes tried every move
    // of both mazes (20 did not always, with libc++), every greedy path was a shortest one,
    // and the largest error was 1.1e-16
    for (int scale : {1, 2}) {
        game_type maze(game_type::maze, 1000, scale);
        auto exact = maze_model(maze, 0.95);
        auto v = exact.optimal_values();
        for (unsigned int seed = 1; seed <= 10; seed++) {
            std::mt19937 prng(seed);
            types::policy memory;
            types::policy random_memory;
            maze_learner learner(1.0, 0.95, 5, 1e-9);
            auto random_agent = [&](const game_type::state_type & s,
                                    const std::vector<unsigned int> & legal) {
                return games::epsilon_greedy<game_type>(random_memory, s, legal, 1.0, prng);
            };
            for (unsigned int i = 0; i < 50; i++) {
                auto r = games::play(maze, random_agent, 100000);
                learner(r.episode, memory, r.terminated);
            }
            // whatever is left in the queue, once there is nothing new to learn
            while (learner.queued() > 0) {
                learner.plan(memory);
            }
            double worst = 0;
            for (int s = 0; s < exact.states; s++) {
                games::position at = {s % maze.columns(), s / maze.columns()};
                for (int a = 0; a < 4; a++) {
                    if (exact.model(s, a).empty()) {
                        continue;
                    }
                    types::state s_t(at);
                    types::action a_t(static_cast<unsigned int>(a));
                    REQUIRE(learner.remembers(s_t, a_t));
                    worst = std::max(worst, std::fabs(memory.value(s_t, a_t) - exact.q(v, s, a)));
                }
            }
            REQUIRE(worst < 1e-6);
            REQUIRE(greedy_path(maze, memory, prng, 1000) == static_cast<std::size_t>(14 * scale));
        }
    }
}

/// verifies Example 8.4: prioritized sweeping needs far fewer updates than Dyna-Q to find a good path
TEST_CASE("prioritized_sweeping: Example 8.4, far fewer updates than Dyna-Q", "[prioritized_sweeping][dyna_maze]")
{
    // ε-greedy (ε = 0.1), α = 1 (the maze is deterministic), γ = 0.95, n = 5 for both, θ = 0.0001,
    // the mazes at scales 1, 2 and 3 (47, 188 and 423 cells). A path is good once it is at
    // most 20% longer than the shortest. Over 10 groups of 10 runs, with libstdc++ and libc++,
    // both found one in every run, and prioritized sweeping needed 7.5 to 332 times fewer updates
    // than Dyna-Q in each run, and 33 to 94 times fewer in all, in each group of 10 runs
    for (int scale : {1, 2, 3}) {
        double dyna = 0;
        double sweeping = 0;
        for (unsigned int seed = 1; seed <= 10; seed++) {
            dyna_learner d(1.0, 0.95, 5, seed + 1000);
            maze_learner p(1.0, 0.95, 5, 0.0001);
            auto by_dyna     = updates_until_good(scale, d, dyna_updates, seed);
            auto by_sweeping = updates_until_good(scale, p, sweeping_updates, seed);
            REQUIRE(by_dyna > 0);
            REQUIRE(by_sweeping > 0);
            REQUIRE(by_sweeping * 3 < by_dyna);
            dyna += static_cast<double>(by_dyna);
            sweeping += static_cast<double>(by_sweeping);
        }
        REQUIRE(sweeping * 10 < dyna);
    }
}
