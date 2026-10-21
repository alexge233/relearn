// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/game.hpp"
#include "../examples/games/random_walk.hpp"
#include "../examples/games/dyna_maze.hpp"
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include "solvers.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of `n_step_sarsa` and `n_step_td` (Sutton & Barto, Sections 6.1, 7.1 and 7.2):
 * - n-step returns worked out by hand, at the end of an episode and in an episode cut short
 * - n = 1 is `sarsa`, n ≥ T is constant-α Monte Carlo, and learning online
 *   (`games::play_online` with a window of n) learns exactly what learning after the episode does
 * - the random walk: Example 6.2, both plots (TD(0) against constant-α Monte Carlo on the right),
 *   and Figure 7.2 (an intermediate n is best)
 * - control: the exact values of the ε-greedy policy in a slippery corridor, and in the
 *   Dyna maze, the first episode (Figure 7.4) and larger n learning faster
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;
using learner = relearn::n_step_sarsa<state,action>;

using walk_type  = games::random_walk;
using walk_types = games::types<walk_type>;
using maze_type  = games::dyna_maze;
using maze_types = games::types<maze_type>;

/**
 * @brief the agent of a prediction task: there is only one action, so it always takes it
 * @return action 0
 */
unsigned int only_action(const unsigned int &,
                         const std::vector<unsigned int> &)
{
    return 0u;
}

/**
 * @brief the root mean square error of the learnt values of a random walk
 * @param walk gives the true values
 * @param memory holds the learnt values, V(s) = Q(s, 0)
 * @return sqrt(Σ_s (V(s) - v(s))² / number of states)
 */
double rms_error(const walk_type & walk,
                 const walk_types::policy & memory)
{
    double sum = 0;
    for (unsigned int s = 1; s <= walk.states(); s++) {
        double error = memory.value(walk_types::state(s), walk_types::action(0)) - walk.true_value(s);
        sum += error * error;
    }
    return std::sqrt(sum / walk.states());
}

/**
 * @brief Example 6.2, the right-hand plot: the RMS error of the 5-state walk after 100 episodes, averaged over 100 runs
 * @param monte_carlo is true for constant-α Monte Carlo, false for TD(0)
 * @param alpha is the learning rate
 * @param seed seeds the walks of the runs (seed * 1000 + run)
 * @return the average RMS error after 100 episodes, every value starting at 0.5
 */
double figure_6_2(bool monte_carlo,
                  double alpha,
                  unsigned int seed)
{
    double error = 0;
    for (unsigned int run = 0; run < 100; run++) {
        walk_type walk(5, 0, 1, seed * 1000 + run);
        walk_types::policy memory;
        for (unsigned int s = 1; s <= 5; s++) {
            memory.update(walk_types::state(s), walk_types::action(0), 0.5);
        }
        relearn::n_step_td<walk_types::state, walk_types::action> td(alpha, 1.0, 1);
        relearn::monte_carlo<walk_types::state, walk_types::action> mc(1.0, false);
        mc.learning_rate = [alpha](std::size_t) { return alpha; };
        for (unsigned int i = 0; i < 100; i++) {
            auto r = games::play(walk, only_action, 100000);
            if (monte_carlo) {
                mc(r.episode, memory, r.terminated);
            }
            else {
                td(r.episode, memory, r.terminated);
            }
        }
        error += rms_error(walk, memory) / 100;
    }
    return error;
}

/**
 * @brief Figure 7.2: the RMS error of the 19-state walk over its first 10 episodes, averaged over runs
 * @param n is the number of steps
 * @param alpha is the learning rate
 * @param runs is the number of runs
 * @param seed seeds the walks of the runs (seed * 1000 + run)
 * @return the RMS error after each of the first 10 episodes, averaged over them and over the runs
 */
double figure_7_2(unsigned int n,
                  double alpha,
                  unsigned int runs,
                  unsigned int seed)
{
    double error = 0;
    for (unsigned int run = 0; run < runs; run++) {
        walk_type walk(19, -1, 1, seed * 1000 + run);
        walk_types::policy memory;
        relearn::n_step_td<walk_types::state, walk_types::action> td(alpha, 1.0, n);
        for (unsigned int i = 0; i < 10; i++) {
            auto r = games::play(walk, only_action, 100000);
            td(r.episode, memory, r.terminated);
            error += rms_error(walk, memory) / (10.0 * runs);
        }
    }
    return error;
}

/**
 * @brief the lowest error of Figure 7.2 for one n, over the learning rates 0.1, 0.2, ... 1
 * @param n is the number of steps
 * @param seed seeds the walks
 * @return the lowest average RMS error, over 50 runs
 */
double best_of_figure_7_2(unsigned int n,
                          unsigned int seed)
{
    double best = std::numeric_limits<double>::infinity();
    for (int i = 1; i <= 10; i++) {
        best = std::min(best, figure_7_2(n, 0.1 * i, 50, seed));
    }
    return best;
}

/**
 * @class corridor
 * @brief the slippery corridor of `sarsa_test.cpp`: 5 cells, 0 ... 4, with the goal at cell 4
 *
 * - actions: 0 moves left (staying put at cell 0), 1 moves right
 * - moving right slips with probability 0.2, and the agent stays put
 * - every move costs -1, and reaching cell 4 ends the episode
 */
class corridor
{
public:
    using state_type  = int;
    using action_type = unsigned int;

    /// @brief create the corridor @param seed seeds the slips
    explicit corridor(unsigned int seed)
    : prng_(seed)
    {}

    /// @brief start a new episode @return cell 0
    state_type reset()
    {
        at_ = 0;
        return at_;
    }

    /// @brief take the move @param a @return the cell reached, the reward and whether the game ended
    games::step_result<state_type> step(const action_type & a)
    {
        games::step_result<state_type> out;
        std::bernoulli_distribution slip(0.2);
        if (a == 1) {
            // moving right succeeds 80% of the time
            if (!slip(prng_)) {
                at_++;
            }
        }
        else if (at_ > 0) {
            // moving left always succeeds, except at the wall
            at_--;
        }
        out.reward = -1;
        out.terminated = (at_ == 4);
        out.next = at_;
        return out;
    }

    /// @return the legal moves: left and right, in every cell
    std::vector<action_type> actions(const state_type &) const
    {
        return {0, 1};
    }

    /**
     * @brief the exact model of the corridor, written from its rules
     * @param gamma is the discount rate
     * @return the model, for the solvers
     */
    static solvers::tabular model(double gamma)
    {
        solvers::tabular m;
        m.states  = 5;
        m.actions = 2;
        m.gamma   = gamma;
        m.model   = [](int s, int a) {
            std::vector<solvers::outcome> outcomes;
            // the goal ends the episode: no action there
            if (s == 4) {
                return outcomes;
            }
            if (a == 1) {
                outcomes.push_back({0.8, s + 1, -1, s + 1 == 4});
                outcomes.push_back({0.2, s, -1, false});
            }
            else {
                outcomes.push_back({1.0, s > 0 ? s - 1 : 0, -1, false});
            }
            return outcomes;
        };
        return m;
    }

private:
    // the cell the agent is in
    int at_ = 0;
    // the random slips
    std::mt19937 prng_;
};

using corridor_types = games::types<corridor>;

/**
 * @brief the largest difference between the learnt values of the corridor and exact ones
 * @param memory is the learnt policy
 * @param m is the exact model
 * @param v are the exact state values the action values are computed from
 * @return the largest |Q(s,a) - exact Q(s,a)| over the cells 0 ... 3 and both moves,
 *         or infinity if a learnt value is not finite
 */
double worst_error(const corridor_types::policy & memory,
                   const solvers::tabular & m,
                   const std::vector<double> & v)
{
    double worst = 0;
    for (int s = 0; s < 4; s++) {
        for (unsigned int a = 0; a < 2; a++) {
            double learnt = memory.value(corridor_types::state(s), corridor_types::action(a));
            if (!std::isfinite(learnt)) {
                return std::numeric_limits<double>::infinity();
            }
            worst = std::max(worst, std::fabs(learnt - m.q(v, s, static_cast<int>(a))));
        }
    }
    return worst;
}

/**
 * @brief learn the corridor online with an ε-greedy agent (ε = 0.1): control
 * @param n is the number of steps of n-step SARSA
 * @param seed seeds the corridor and the agent
 * @return the learnt policy, after 20000 episodes, with α = 1/k^0.6 for the k-th update of each pair
 */
corridor_types::policy learn_corridor(unsigned int n,
                                      unsigned int seed)
{
    corridor game(seed);
    std::mt19937 prng(seed + 100);
    corridor_types::policy memory;
    relearn::n_step_sarsa<corridor_types::state, corridor_types::action> sarsa(0.1, 0.9, n);
    sarsa.learning_rate = relearn::polynomial_decay<>(1.0, 0.6);
    // the ε-greedy agent, greedy on what it has learnt so far
    auto agent = [&](const int & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<corridor>(memory, s, legal, 0.1, prng);
    };
    for (int k = 1; k <= 20000; k++) {
        games::play_online(game, agent, sarsa, memory, 1000, n);
    }
    return memory;
}

/**
 * @brief the mean moves of episodes 2 to 10 in the Dyna maze, over 30 runs
 * @param n is the number of steps of n-step SARSA
 * @param seed seeds the runs (seed * 1000 + run)
 * @return the mean moves (the first episode is a random walk whatever n is, so it is left out)
 *
 * ε-greedy with ε = 0.1, α = 0.1, γ = 0.95, learning after every move, as in Figure 8.2
 */
double early_moves(unsigned int n,
                   unsigned int seed)
{
    double moves = 0;
    for (unsigned int run = 0; run < 30; run++) {
        std::mt19937 prng(seed * 1000 + run);
        maze_type maze;
        maze_types::policy memory;
        relearn::n_step_sarsa<maze_types::state, maze_types::action> sarsa(0.1, 0.95, n);
        auto agent = [&](const maze_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<maze_type>(memory, s, legal, 0.1, prng);
        };
        for (unsigned int i = 1; i <= 10; i++) {
            auto r = games::play_online(maze, agent, sarsa, memory, 100000, n);
            if (i > 1) {
                moves += static_cast<double>(r.steps) / (9 * 30);
            }
        }
    }
    return moves;
}
} // namespace

/// verifies n-step returns worked out by hand, in the order of the book's algorithm
TEST_CASE("n_step_sarsa: n-step returns worked out by hand", "[n_step]")
{
    // n = 2, γ = 0.5, α = 0.5, and Q(s2, c) = 4 before the episode, which ends after 3 steps (T = 3):
    //    τ = 0: G = r0 + γ r1 + γ² Q(s2, c) = 1 + 0 + 0.25 * 4 = 2,  Q(s0, a) = 0 + 0.5 (2 - 0) = 1
    //    τ = 1: G = r1 + γ r2 = 0 + 0.5 * 2 = 1 (τ + n = T: nothing after it), Q(s1, b) = 0.5
    //    τ = 2: G = r2 = 2,  Q(s2, c) = 4 + 0.5 (2 - 4) = 3
    // Learnt backwards instead, τ = 0 would bootstrap from Q(s2, c) = 3, and give 0.875
    learner sarsa(0.5, 0.5, 2);
    policy memory;
    memory.update(state("s2"), action("c"), 4);
    auto three = episode {{state(1, "s0"), action("a")},
                          {state(0, "s1"), action("b")},
                          {state(2, "s2"), action("c")}};
    REQUIRE(sarsa.n_step_return(three, 0, memory) == Approx(2));
    REQUIRE(sarsa.n_step_return(three, 1, memory) == Approx(1));
    REQUIRE(sarsa.n_step_return(three, 2, memory) == Approx(2));
    sarsa(three, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(1));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(0.5));
    REQUIRE(memory.value(state("s2"), action("c")) == Approx(3));
}

/// verifies that each update uses the values as they are at that point of the episode
TEST_CASE("n_step_sarsa: a later update bootstraps from a value learnt earlier in the episode", "[n_step]")
{
    // n = 2, γ = 1, α = 0.5: (x, a) is visited at steps 0 and 3, T = 4
    //    τ = 0: G = r0 + r1 + Q(w, d) = 1 + 0 + 0 = 1,   Q(x, a) = 0.5
    //    τ = 1: G = r1 + r2 + Q(x, a) = 0 + 0 + 0.5,     Q(y, b) = 0.25 (with Q(x, a) from before the episode: 0)
    //    τ = 2: G = r2 + r3 = 0,                          Q(w, d) = 0
    //    τ = 3: G = r3 = 0,                               Q(x, a) = 0.5 + 0.5 (0 - 0.5) = 0.25
    learner sarsa(0.5, 1.0, 2);
    policy memory;
    sarsa(episode {{state(1, "x"), action("a")},
                   {state(0, "y"), action("b")},
                   {state(0, "w"), action("d")},
                   {state(0, "x"), action("a")}}, memory);
    REQUIRE(memory.value(state("y"), action("b")) == Approx(0.25));
    REQUIRE(memory.value(state("w"), action("d")) == Approx(0));
    REQUIRE(memory.value(state("x"), action("a")) == Approx(0.25));
}

/// verifies an episode cut short: only the steps whose n rewards are known are learnt
TEST_CASE("n_step_sarsa: an episode cut short learns the steps whose n rewards are known", "[n_step]")
{
    // four steps, then the state reached and the action chosen next, (s4, e), worth 8.
    // n = 2, γ = 0.5, α = 1, so each value is its return:
    //    τ = 0: r0 + γ r1 + γ² Q(s2, c) = 1 + 0.5 * 2 + 0 = 2
    //    τ = 1: r1 + γ r2 + γ² Q(s3, d) = 2 + 0.5 * 3 + 0 = 3.5
    //    τ = 2: r2 + γ r3 + γ² Q(s4, e) = 3 + 0.5 * 4 + 0.25 * 8 = 7
    //    τ = 3 needs r4, which is not known: it is not learnt (the reward of (s4, e) is not a reward)
    learner sarsa(1.0, 0.5, 2);
    policy memory;
    memory.update(state("s4"), action("e"), 8);
    auto cut = episode {{state(1, "s0"), action("a")},
                        {state(2, "s1"), action("b")},
                        {state(3, "s2"), action("c")},
                        {state(4, "s3"), action("d")},
                        {state(100, "s4"), action("e")}};
    sarsa(cut, memory, false);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(2));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(3.5));
    REQUIRE(memory.value(state("s2"), action("c")) == Approx(7));
    REQUIRE(memory.actions(state("s3")).empty());
    REQUIRE(memory.value(state("s4"), action("e")) == Approx(8));
    // an episode cut short before n steps learns nothing at all
    policy untouched;
    learner four(1.0, 0.5, 4);
    four(episode {{state(1, "s0"), action("a")},
                  {state(2, "s1"), action("b")},
                  {state(0, "s2"), action("c")}}, untouched, false);
    REQUIRE(untouched.actions(state("s0")).empty());
    REQUIRE(untouched.actions(state("s1")).empty());
}

/// verifies that with n = 1, n-step SARSA is SARSA, to the last bit
TEST_CASE("n_step_sarsa: n = 1 is sarsa", "[n_step]")
{
    // the same corridor episodes, played at random, learnt by both
    corridor game(3);
    std::mt19937 prng(4);
    auto agent = [&](const int &,
                     const std::vector<unsigned int> & legal) {
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        return legal[pick(prng)];
    };
    relearn::sarsa<corridor_types::state, corridor_types::action> one_step(0.3, 0.9);
    relearn::n_step_sarsa<corridor_types::state, corridor_types::action> n_step(0.3, 0.9, 1);
    corridor_types::policy one_memory;
    corridor_types::policy n_memory;
    for (int i = 0; i < 200; i++) {
        // some episodes are cut short, after 5 moves
        auto r = games::play(game, agent, i % 2 == 0 ? 5 : 1000);
        one_step(r.episode, one_memory, r.terminated);
        n_step(r.episode, n_memory, r.terminated);
    }
    for (int s = 0; s < 4; s++) {
        for (unsigned int a = 0; a < 2; a++) {
            REQUIRE(n_memory.value(corridor_types::state(s), corridor_types::action(a)) ==
                    one_memory.value(corridor_types::state(s), corridor_types::action(a)));
        }
    }
}

/// verifies that when n reaches past the end of every episode, n-step SARSA is constant-α Monte Carlo
TEST_CASE("n_step_sarsa: n >= T is constant-alpha Monte Carlo", "[n_step][monte_carlo]")
{
    // γ = 0.9, α = 0.2, episodes that visit each pair once: the return to the end, with nothing after it
    learner sarsa(0.2, 0.9, 10);
    relearn::monte_carlo<state,action> mc(0.9, false);
    mc.learning_rate = [](std::size_t) { return 0.2; };
    policy n_memory;
    policy mc_memory;
    for (int i = 0; i < 3; i++) {
        auto e = episode {{state(1, "s0"), action("a")},
                          {state(-2, "s1"), action("b")},
                          {state(3 + i, "s2"), action("c")}};
        sarsa(e, n_memory);
        mc(e, mc_memory);
    }
    for (auto name : {"s0", "s1", "s2"}) {
        auto a = mc_memory.actions(state(name)).begin()->first;
        REQUIRE(n_memory.value(state(name), a) == Approx(mc_memory.value(state(name), a)));
    }
}

/// verifies that learning online, after every step, learns exactly what learning after the episode does
TEST_CASE("n_step_td: online with a window of n is the same as after the episode", "[n_step][random_walk]")
{
    // prediction: the agent's choice does not depend on the values, so both see the same walks.
    // Half of the walks are cut short, after 20 steps
    for (unsigned int n : {1u, 2u, 4u, 8u}) {
        walk_type online_walk(19, -1, 1, 5);
        walk_type offline_walk(19, -1, 1, 5);
        walk_types::policy online_memory;
        walk_types::policy offline_memory;
        relearn::n_step_td<walk_types::state, walk_types::action> online(0.4, 1.0, n);
        relearn::n_step_td<walk_types::state, walk_types::action> offline(0.4, 1.0, n);
        for (int i = 0; i < 50; i++) {
            std::size_t limit = i % 2 == 0 ? 20 : 100000;
            games::play_online(online_walk, only_action, online, online_memory, limit, n);
            auto r = games::play(offline_walk, only_action, limit);
            offline(r.episode, offline_memory, r.terminated);
        }
        for (unsigned int s = 1; s <= 19; s++) {
            INFO("n = " << n << ", state " << s);
            REQUIRE(online_memory.value(walk_types::state(s), walk_types::action(0)) ==
                    offline_memory.value(walk_types::state(s), walk_types::action(0)));
        }
    }
}

/// verifies the constructor, and brace initialisation
TEST_CASE("n_step_sarsa: constructor and brace initialisation", "[n_step]")
{
    learner defaults;
    REQUIRE(defaults.alpha == Approx(0.9));
    REQUIRE(defaults.gamma == Approx(0.9));
    REQUIRE(defaults.n == 1);
    auto four = learner{0.1, 1.0, 4};
    REQUIRE(four.alpha == Approx(0.1));
    REQUIRE(four.gamma == Approx(1));
    REQUIRE(four.n == 4);
    relearn::n_step_td<state,action> td(0.2, 0.5, 3);
    REQUIRE(td.n == 3);
}

/// verifies Example 6.2: TD(0) learns the true values of the 5-state walk, 1/6 ... 5/6
TEST_CASE("n_step_td: TD(0) predicts the 5-state random walk (Example 6.2)", "[n_step][random_walk]")
{
    // α = 0.01, 5000 walks: over 10 seeds, with libstdc++ and with libc++, the largest error was 0.006 to 0.039
    walk_type walk(5, 0, 1, 1);
    walk_types::policy memory;
    relearn::n_step_td<walk_types::state, walk_types::action> td(0.01, 1.0, 1);
    for (int i = 0; i < 5000; i++) {
        auto r = games::play(walk, only_action, 100000);
        td(r.episode, memory, r.terminated);
    }
    for (unsigned int s = 1; s <= 5; s++) {
        INFO("state " << s);
        REQUIRE(memory.value(walk_types::state(s), walk_types::action(0)) ==
                Approx(walk.true_value(s)).margin(0.06));
    }
}

/// verifies Example 6.2's right-hand plot: on the 5-state walk, TD(0) learns faster than constant-α Monte Carlo
TEST_CASE("n_step_td: TD(0) beats constant-alpha Monte Carlo (Example 6.2)", "[n_step][random_walk][monte_carlo]")
{
    // after 100 walks, averaged over 100 runs. Over 10 seeds, with libstdc++ and with libc++, the RMS error was
    // TD(0) α = 0.05: 0.033 to 0.041, α = 0.1: 0.054 to 0.062, and constant-α MC α = 0.01: 0.092 to 0.100,
    // α = 0.02: 0.076 to 0.089, α = 0.03: 0.086 to 0.104, α = 0.04: 0.098 to 0.119
    double td = std::min(figure_6_2(false, 0.05, 1), figure_6_2(false, 0.1, 1));
    double mc = std::numeric_limits<double>::infinity();
    for (double alpha : {0.01, 0.02, 0.03, 0.04}) {
        mc = std::min(mc, figure_6_2(true, alpha, 1));
    }
    INFO("TD(0) " << td << ", Monte Carlo " << mc);
    REQUIRE(td < 0.05);
    REQUIRE(mc > 0.065);
    REQUIRE(td < mc - 0.025);
}

/// verifies Figure 7.2: on the 19-state walk, an intermediate n learns best in the first 10 episodes
TEST_CASE("n_step_td: an intermediate n is best (Figure 7.2)", "[n_step][random_walk]")
{
    // each n at its best α (0.1 ... 1), over 50 runs. Over 10 seeds, with libstdc++ and with libc++, the errors were
    // n = 1: 0.341 to 0.348, n = 2: 0.270 to 0.282, n = 4: 0.255 to 0.270, n = 8: 0.269 to 0.287,
    // n = 32: 0.352 to 0.370, and n = 4 was the best by at least 0.01 every time
    std::vector<unsigned int> ns = {1, 2, 4, 8, 32};
    std::vector<double> best;
    for (auto n : ns) {
        best.push_back(best_of_figure_7_2(n, 1));
    }
    INFO("n = 1: " << best[0] << ", 2: " << best[1] << ", 4: " << best[2]
         << ", 8: " << best[3] << ", 32: " << best[4]);
    // n = 4 is the best, as in the book, with about 0.25; n = 1 and n = 32 are much worse
    REQUIRE(best[2] < best[1]);
    REQUIRE(best[2] < best[3]);
    REQUIRE(best[2] < best[0] - 0.05);
    REQUIRE(best[2] < best[4] - 0.05);
    REQUIRE(best[2] == Approx(0.25).margin(0.03));
}

/// verifies n-step SARSA control: with a constant ε it learns the exact values of the ε-greedy policy
TEST_CASE("n_step_sarsa: learns q of the epsilon-greedy policy in the corridor", "[n_step][convergence]")
{
    // the best move is right in every cell, so the ε-greedy policy (ε = 0.1) moves right with 0.95, left 0.05
    const double gamma = 0.9;
    auto exact = corridor::model(gamma);
    std::vector<std::vector<double>> pi(5, {0.05, 0.95});
    auto v_epsilon = exact.policy_values(pi);
    // over 10 seeds, with libstdc++ and with libc++, the worst errors were
    // n = 1: 0.016 to 0.077, n = 2: 0.034 to 0.120, n = 4: 0.027 to 0.152
    for (unsigned int n : {1u, 2u, 4u}) {
        auto memory = learn_corridor(n, 1);
        INFO("n = " << n);
        REQUIRE(worst_error(memory, exact, v_epsilon) < 0.25);
    }
}

/// verifies Figure 7.4: after the first episode, n-step SARSA has learnt the last n moves, one-step SARSA only the last
TEST_CASE("n_step_sarsa: the first episode learns its last n moves (Figure 7.4)", "[n_step][dyna_maze]")
{
    // the maze is worth 0 everywhere except the goal, so only the moves whose n-step return
    // reaches the goal get a value: the last n moves, that is the different pairs among them
    for (unsigned int n : {1u, 4u, 10u}) {
        std::mt19937 prng(n);
        maze_type maze;
        maze_types::policy memory;
        relearn::n_step_sarsa<maze_types::state, maze_types::action> sarsa(0.1, 0.95, n);
        auto agent = [&](const maze_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<maze_type>(memory, s, legal, 0.1, prng);
        };
        auto r = games::play_online(maze, agent, sarsa, memory, 100000, n);
        REQUIRE(r.terminated);
        std::set<std::pair<int,int>> last_moves;
        for (std::size_t i = r.episode.size() - n; i < r.episode.size(); i++) {
            auto p = r.episode[i].state.trait();
            last_moves.insert({p.y * maze_type::width + p.x, static_cast<int>(r.episode[i].action.trait())});
        }
        // count the pairs with a value, over the whole maze
        std::size_t valued = 0;
        for (int x = 0; x < maze_type::width; x++) {
            for (int y = 0; y < maze_type::height; y++) {
                for (unsigned int a = 0; a < 4; a++) {
                    double q = memory.value(maze_types::state(games::position{x, y}), maze_types::action(a));
                    valued += q > 0;
                    if (q > 0) {
                        REQUIRE(last_moves.count({y * maze_type::width + x, static_cast<int>(a)}) == 1);
                    }
                }
            }
        }
        INFO("n = " << n);
        REQUIRE(valued == last_moves.size());
    }
}

/// verifies that a larger n learns the Dyna maze in fewer episodes (Section 7.2)
TEST_CASE("n_step_sarsa: a larger n learns the maze faster in the early episodes", "[n_step][dyna_maze]")
{
    // the mean moves of episodes 2 to 10, over 30 runs. Over 10 seeds, with libstdc++ and with libc++,
    // they were n = 1: 276 to 350, n = 4: 117 to 175, n = 8: 96 to 168 (at most 0.61 of n = 1)
    double one   = early_moves(1, 1);
    double four  = early_moves(4, 1);
    double eight = early_moves(8, 1);
    INFO("n = 1: " << one << ", n = 4: " << four << ", n = 8: " << eight);
    REQUIRE(four < 0.75 * one);
    REQUIRE(eight < 0.75 * one);
}

/********************************************************************************
 ***  n-step Expected SARSA, off-policy n-step SARSA, tree backup and Q(σ)
 ***  (Sutton & Barto, Sections 7.2 to 7.6)
 ********************************************************************************/

namespace
{
using distribution = std::vector<std::pair<action,double>>;

/**
 * @brief π for the worked examples: in s1, b and d with 0.5 each; in s2, c with 0.25 and e with 0.75;
 *        elsewhere, the action of the episode for sure
 * @param s is the state
 * @return π(· | s)
 */
distribution worked_pi(const state & s)
{
    if (s.trait() == "s1") {
        return {{action("b"), 0.5}, {action("d"), 0.5}};
    }
    if (s.trait() == "s2") {
        return {{action("c"), 0.25}, {action("e"), 0.75}};
    }
    return {{action("a"), 1.0}};
}

/// @brief b for the worked examples: every action with probability 0.5 @return 0.5
double worked_b(const state &,
                const action &)
{
    return 0.5;
}

/// @return the episode of the worked examples: (s0, a, r = 1), (s1, b, r = 0), (s2, c, r = 2), which ends
episode worked_episode()
{
    return episode {{state(1, "s0"), action("a")},
                    {state(0, "s1"), action("b")},
                    {state(2, "s2"), action("c")}};
}

/// @return the values before the worked examples: Q(s1, d) = 2, Q(s2, c) = 4, Q(s2, e) = 8, and 0 for the rest
policy worked_values()
{
    policy memory;
    memory.update(state("s1"), action("d"), 2);
    memory.update(state("s2"), action("c"), 4);
    memory.update(state("s2"), action("e"), 8);
    return memory;
}

/**
 * @brief random episodes of the corridor, played by an agent that moves right with 0.7, left with 0.3
 * @param count is the number of episodes
 * @param seed seeds the corridor and the agent
 * @return the episodes, every other one cut short after 4 moves, with whether each really ended
 */
std::vector<std::pair<corridor_types::episode,bool>> corridor_episodes(int count,
                                                                       unsigned int seed)
{
    corridor game(seed);
    std::mt19937 prng(seed + 1);
    std::bernoulli_distribution right(0.7);
    auto agent = [&](const int &,
                     const std::vector<unsigned int> &) {
        return right(prng) ? 1u : 0u;
    };
    std::vector<std::pair<corridor_types::episode,bool>> episodes;
    for (int i = 0; i < count; i++) {
        auto r = games::play(game, agent, i % 2 == 0 ? 4 : 1000);
        episodes.push_back({r.episode, r.terminated});
    }
    return episodes;
}

/// @brief b of the corridor's agent above @return 0.7 for right, 0.3 for left
double corridor_b(const corridor_types::state &,
                  const corridor_types::action & a)
{
    return a.trait() == 1 ? 0.7 : 0.3;
}

/// @brief a fixed π for the corridor: right with 0.6, left with 0.4 @return π(· | s)
std::vector<std::pair<corridor_types::action,double>> corridor_pi(const corridor_types::state &)
{
    return {{corridor_types::action(0), 0.4}, {corridor_types::action(1), 0.6}};
}

/**
 * @brief the largest difference between two learnt policies of the corridor
 * @return the largest |Q1(s,a) - Q2(s,a)| over the cells 0 ... 3 and both moves
 */
double largest_difference(const corridor_types::policy & one,
                          const corridor_types::policy & two)
{
    double largest = 0;
    for (int s = 0; s < 4; s++) {
        for (unsigned int a = 0; a < 2; a++) {
            largest = std::max(largest, std::fabs(one.value(corridor_types::state(s), corridor_types::action(a)) -
                                                  two.value(corridor_types::state(s), corridor_types::action(a))));
        }
    }
    return largest;
}

/**
 * @brief learn the corridor online, playing with @param play_right (the probability of moving right),
 *        with α = α0/k^0.6 for the k-th update of each pair, γ = 0.9, for 20000 episodes
 * @param learner is the learner, to which @param prepare gives π (and b) on the values being learnt
 * @param alpha_0 is α0, the first step size of each pair
 * @param n is its number of steps, the window of `games::play_online`
 * @param seed seeds the corridor and the agent
 * @return the learnt policy
 */
template <class learner_type,
          class preparer>
corridor_types::policy learn_corridor_with(learner_type learner,
                                           preparer && prepare,
                                           double play_right,
                                           double alpha_0,
                                           unsigned int n,
                                           unsigned int seed)
{
    corridor game(seed);
    std::mt19937 prng(seed + 100);
    std::bernoulli_distribution right(play_right);
    corridor_types::policy memory;
    learner.learning_rate = relearn::polynomial_decay<>(alpha_0, 0.6);
    prepare(learner, memory);
    auto agent = [&](const int &,
                     const std::vector<unsigned int> &) {
        return right(prng) ? 1u : 0u;
    };
    for (int k = 1; k <= 20000; k++) {
        games::play_online(game, agent, learner, memory, 1000, n);
    }
    return memory;
}
} // namespace

/// verifies n-step Expected SARSA's returns worked out by hand, on-policy and off-policy
TEST_CASE("n_step_expected_sarsa: n-step returns worked out by hand", "[n_step][expected_sarsa]")
{
    // n = 2, γ = 0.5, α = 0.5; V̄(s2) = 0.25 * 4 + 0.75 * 8 = 7 before the episode
    //    τ = 0: G = r0 + γ r1 + γ² V̄(s2) = 1 + 0 + 0.25 * 7 = 2.75,  Q(s0, a) = 0.5 * 2.75 = 1.375
    //    τ = 1: G = r1 + γ r2 = 0 + 0.5 * 2 = 1 (the episode ends),  Q(s1, b) = 0.5
    //    τ = 2: G = r2 = 2,                                          Q(s2, c) = 4 + 0.5 (2 - 4) = 3
    relearn::n_step_expected_sarsa<state,action> on(0.5, 0.5, 2, worked_pi);
    auto memory = worked_values();
    REQUIRE(on.expected_value(state("s2"), memory) == Approx(7));
    REQUIRE(on.n_step_return(worked_episode(), 0, memory) == Approx(2.75));
    REQUIRE(on.importance_ratio(worked_episode(), 0, memory) == Approx(1));
    on(worked_episode(), memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(1.375));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(0.5));
    REQUIRE(memory.value(state("s2"), action("c")) == Approx(3));
    // off-policy, b = 0.5 everywhere: ρ_{τ+1:τ+n-1} has n - 1 = 1 factor
    //    τ = 0: ρ = π(b | s1) / b = 0.5 / 0.5 = 1,      Q(s0, a) = 0.5 * 1 * 2.75 = 1.375
    //    τ = 1: ρ = π(c | s2) / b = 0.25 / 0.5 = 0.5,   Q(s1, b) = 0.5 * 0.5 * 1 = 0.25
    //    τ = 2: ρ = 1 (no action after a_τ in the episode),  Q(s2, c) = 3
    relearn::n_step_expected_sarsa<state,action> off(0.5, 0.5, 2, worked_pi);
    off.behaviour = worked_b;
    auto off_memory = worked_values();
    REQUIRE(off.importance_ratio(worked_episode(), 1, off_memory) == Approx(0.5));
    REQUIRE(off.importance_ratio(worked_episode(), 2, off_memory) == Approx(1));
    off(worked_episode(), off_memory);
    REQUIRE(off_memory.value(state("s0"), action("a")) == Approx(1.375));
    REQUIRE(off_memory.value(state("s1"), action("b")) == Approx(0.25));
    REQUIRE(off_memory.value(state("s2"), action("c")) == Approx(3));
}

/// verifies off-policy n-step SARSA's ratios and updates worked out by hand
TEST_CASE("off_policy_n_step_sarsa: ratios worked out by hand", "[n_step]")
{
    // n = 2, γ = 0.5, α = 0.5, b = 0.5 everywhere; the return is n-step SARSA's:
    //    τ = 0: G = r0 + γ r1 + γ² Q(s2, c) = 1 + 0 + 0.25 * 4 = 2,
    //           ρ_{1:2} = π(b | s1) / b * π(c | s2) / b = 1 * 0.5 = 0.5,  Q(s0, a) = 0.5 * 0.5 * 2 = 0.5
    //    τ = 1: G = r1 + γ r2 = 1, ρ_{2:2} = 0.5 (no factor after the last step),  Q(s1, b) = 0.5 * 0.5 * 1 = 0.25
    //    τ = 2: G = 2, ρ = 1 (no factor at all),  Q(s2, c) = 4 + 0.5 (2 - 4) = 3
    relearn::off_policy_n_step_sarsa<state,action> sarsa(0.5, 0.5, 2, worked_pi, worked_b);
    auto memory = worked_values();
    REQUIRE(sarsa.n_step_return(worked_episode(), 0, memory) == Approx(2));
    REQUIRE(sarsa.importance_ratio(worked_episode(), 0, memory) == Approx(0.5));
    REQUIRE(sarsa.importance_ratio(worked_episode(), 1, memory) == Approx(0.5));
    REQUIRE(sarsa.importance_ratio(worked_episode(), 2, memory) == Approx(1));
    sarsa(worked_episode(), memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(0.5));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(0.25));
    REQUIRE(memory.value(state("s2"), action("c")) == Approx(3));
}

/// verifies that a step π would never have played teaches nothing, and is not counted as an update
TEST_CASE("off_policy_n_step_sarsa: a ratio of 0 is not an update", "[n_step]")
{
    // π takes only a, b takes a and z with 0.5 each: the step before z has ρ = 0
    auto only_a = [](const state &) { return distribution {{action("a"), 1.0}}; };
    relearn::off_policy_n_step_sarsa<state,action> sarsa(0.5, 1.0, 1, only_a, worked_b);
    sarsa.learning_rate = relearn::polynomial_decay<>(1.0, 1.0);
    policy memory;
    sarsa(episode {{state(1, "x"), action("a")},
                   {state(5, "y"), action("z")}}, memory);
    // (x, a): ρ = π(z | y) / b = 0, not learnt nor counted; (y, z): ρ = 1 (no action after it), Q = 5
    REQUIRE(memory.actions(state("x")).empty());
    REQUIRE(sarsa.visits.count(state("x"), action("a")) == 0);
    REQUIRE(memory.value(state("y"), action("z")) == Approx(5));
}

/// verifies that with b = π, off-policy n-step SARSA is n-step SARSA, to the last bit
TEST_CASE("off_policy_n_step_sarsa: with b = pi it is n_step_sarsa", "[n_step]")
{
    auto b_is_pi = [](const corridor_types::state & s, const corridor_types::action & a) {
        return a.trait() == 1 ? corridor_pi(s)[1].second : corridor_pi(s)[0].second;
    };
    for (unsigned int n : {1u, 2u, 4u}) {
        relearn::off_policy_n_step_sarsa<corridor_types::state, corridor_types::action> off(0.3, 0.9, n, corridor_pi, b_is_pi);
        relearn::n_step_sarsa<corridor_types::state, corridor_types::action> on(0.3, 0.9, n);
        corridor_types::policy off_memory;
        corridor_types::policy on_memory;
        for (const auto & e : corridor_episodes(200, n)) {
            off(e.first, off_memory, e.second);
            on(e.first, on_memory, e.second);
        }
        INFO("n = " << n);
        REQUIRE(largest_difference(off_memory, on_memory) == 0);
    }
}

/// verifies tree backup's return worked out by hand
TEST_CASE("n_step_tree_backup: n-step returns worked out by hand", "[n_step][tree_backup]")
{
    // n = 2, γ = 0.5, α = 0.5:
    //    τ = 0: G_{1:2} = r1 + γ V̄(s2) = 0 + 0.5 * 7 = 3.5, then
    //           G_{0:2} = r0 + γ π(d | s1) Q(s1, d) + γ π(b | s1) G_{1:2} = 1 + 0.5 * 1 + 0.25 * 3.5 = 2.375,
    //           Q(s0, a) = 0.5 * 2.375 = 1.1875
    //    τ = 1: G_{2:3} = r2 = 2 (the episode ends), then
    //           G_{1:3} = r1 + γ π(e | s2) Q(s2, e) + γ π(c | s2) G_{2:3} = 0 + 0.5 * 6 + 0.5 * 0.25 * 2 = 3.25,
    //           Q(s1, b) = 0.5 * 3.25 = 1.625
    //    τ = 2: G = r2 = 2,  Q(s2, c) = 4 + 0.5 (2 - 4) = 3
    relearn::n_step_tree_backup<state,action> tree(0.5, 0.5, 2, worked_pi);
    auto memory = worked_values();
    REQUIRE(tree.n_step_return(worked_episode(), 0, memory) == Approx(2.375));
    REQUIRE(tree.n_step_return(worked_episode(), 1, memory) == Approx(3.25));
    tree(worked_episode(), memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(1.1875));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(1.625));
    REQUIRE(memory.value(state("s2"), action("c")) == Approx(3));
}

/// verifies that with a greedy π, tree backup follows the rewards for as long as the actions were greedy
TEST_CASE("n_step_tree_backup: a greedy pi cuts the return at the first other action", "[n_step][tree_backup]")
{
    // no π given: greedy on the values stored. In y, the best action is g (worth 10), and the episode took z.
    // n = 2, γ = 1, α = 1, so Q(x, a) is its return: G = r0 + max_a Q(y, a) = 1 + 10 = 11,
    // whatever came after z (its reward of 100 is never used). The episode is cut short at w: only τ = 0 is learnt
    relearn::n_step_tree_backup<state,action> tree(1.0, 1.0, 2);
    policy memory;
    memory.update(state("y"), action("g"), 10);
    memory.update(state("y"), action("z"), 0);
    tree(episode {{state(1, "x"), action("a")},
                  {state(100, "y"), action("z")},
                  {state(0, "w"), action("d")}}, memory, false);
    REQUIRE(memory.value(state("x"), action("a")) == Approx(11));
    // and when the greedy action was taken, the return follows it: r0 + r1 + max_a Q(w, a) = 1 + 100 + 0
    policy greedy_taken;
    greedy_taken.update(state("y"), action("g"), 10);
    tree(episode {{state(1, "x"), action("a")},
                  {state(100, "y"), action("g")},
                  {state(0, "w"), action("d")}}, greedy_taken, false);
    REQUIRE(greedy_taken.value(state("x"), action("a")) == Approx(101));
}

/// verifies the greedy π used when none is given: the actions of highest value share the probability
TEST_CASE("n-step family: the greedy pi shares its probability between tied actions", "[n_step]")
{
    // in s, g and h are both worth 4, and z is worth 1: π(g | s) = π(h | s) = 0.5, and V̄(s) = 4
    policy memory;
    memory.update(state("s"), action("g"), 4);
    memory.update(state("s"), action("h"), 4);
    memory.update(state("s"), action("z"), 1);
    auto pi = relearn::target_policy(std::function<distribution(const state &)>(), state("s"), memory);
    REQUIRE(pi.size() == 2);
    REQUIRE(relearn::probability_of(pi, action("g")) == Approx(0.5));
    REQUIRE(relearn::probability_of(pi, action("h")) == Approx(0.5));
    REQUIRE(relearn::probability_of(pi, action("z")) == 0);
    relearn::n_step_expected_sarsa<state,action> expected(0.5, 1.0, 1);
    REQUIRE(expected.expected_value(state("s"), memory) == Approx(4));
    // a state never seen has no actions stored: π is empty, and V̄ is 0
    REQUIRE(relearn::target_policy(std::function<distribution(const state &)>(), state("new"), memory).empty());
    REQUIRE(expected.expected_value(state("new"), memory) == 0);
}

/// verifies Q(σ)'s returns worked out by hand, on-policy and off-policy
TEST_CASE("n_step_q_sigma: n-step returns worked out by hand", "[n_step][q_sigma]")
{
    // n = 2, γ = 0.5, α = 0.5, σ = 0.5, on-policy (ρ = 1):
    //    τ = 0: G_{2:2} = Q(s2, c) = 4, then
    //           G_{1:2} = r1 + γ V̄(s2) + γ (σ + (1 - σ) 0.25)(G_{2:2} - Q(s2, c)) = 0 + 3.5 + 0 = 3.5,
    //           G_{0:2} = r0 + γ V̄(s1) + γ (σ + (1 - σ) 0.5)(G_{1:2} - Q(s1, b)) = 1 + 0.5 * 1 + 0.5 * 0.75 * 3.5 = 2.8125,
    //           Q(s0, a) = 0.5 * 2.8125 = 1.40625
    //    τ = 1: G_{2:3} = r2 = 2, then G_{1:3} = 0 + 3.5 + 0.5 * 0.625 * (2 - 4) = 2.875,  Q(s1, b) = 1.4375
    //    τ = 2: G = r2 = 2,  Q(s2, c) = 3
    relearn::n_step_q_sigma<state,action> q_sigma(0.5, 0.5, 2, 0.5, worked_pi);
    auto memory = worked_values();
    REQUIRE(q_sigma.n_step_return(worked_episode(), 0, memory) == Approx(2.8125));
    q_sigma(worked_episode(), memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(1.40625));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(1.4375));
    REQUIRE(memory.value(state("s2"), action("c")) == Approx(3));
    // σ = 1, off-policy with b(b | s1) = 0.25 and 0.5 elsewhere: ρ = 0.5 / 0.25 = 2 in s1, 0.25 / 0.5 = 0.5 in s2
    //    τ = 0: G_{1:2} = 3.5 + 0.5 * 0.5 * (4 - 4) = 3.5,  G_{0:2} = 1 + 0.5 * 1 + 0.5 * 2 * (3.5 - 0) = 5
    //    (on-policy, ρ = 1, it would be 3.25)
    relearn::n_step_q_sigma<state,action> sampled(0.5, 0.5, 2, 1.0, worked_pi);
    sampled.behaviour = [](const state & s, const action & a) {
        return (s.trait() == "s1" && a.trait() == "b") ? 0.25 : 0.5;
    };
    REQUIRE(sampled.n_step_return(worked_episode(), 0, worked_values()) == Approx(5));
    // σ chosen per step: σ = 1 in s1 and 0 in s2 gives, for τ = 0, G_{1:2} = 3.5 and
    // G_{0:2} = 1 + 0.5 * 1 + 0.5 * 1 * 3.5 = 3.25 as well (the σ of s2 meets G - Q = 0)
    relearn::n_step_q_sigma<state,action> mixed(0.5, 0.5, 2, 0.5, worked_pi);
    mixed.sigma_of = [](const state & s, const action &) { return s.trait() == "s1" ? 1.0 : 0.0; };
    REQUIRE(mixed.n_step_return(worked_episode(), 0, worked_values()) == Approx(3.25));
}

/// verifies that σ = 0 is tree backup, and that n = 1 is Expected SARSA whatever σ is, to rounding
TEST_CASE("n_step_q_sigma: sigma = 0 is tree backup, n = 1 is expected_sarsa", "[n_step][q_sigma][tree_backup]")
{
    // the same corridor episodes (half of them cut short), a fixed π, learnt by each
    for (unsigned int n : {1u, 2u, 4u}) {
        relearn::n_step_q_sigma<corridor_types::state, corridor_types::action> q_zero(0.3, 0.9, n, 0.0, corridor_pi);
        relearn::n_step_tree_backup<corridor_types::state, corridor_types::action> tree(0.3, 0.9, n, corridor_pi);
        corridor_types::policy zero_memory;
        corridor_types::policy tree_memory;
        for (const auto & e : corridor_episodes(200, n)) {
            q_zero(e.first, zero_memory, e.second);
            tree(e.first, tree_memory, e.second);
        }
        INFO("n = " << n);
        REQUIRE(largest_difference(zero_memory, tree_memory) < 1e-12);
    }
    for (double sigma : {0.0, 0.5, 1.0}) {
        relearn::n_step_q_sigma<corridor_types::state, corridor_types::action> q_sigma(0.3, 0.9, 1, sigma, corridor_pi);
        q_sigma.behaviour = corridor_b;
        relearn::expected_sarsa<corridor_types::state, corridor_types::action> expected(0.3, 0.9, corridor_pi);
        relearn::n_step_expected_sarsa<corridor_types::state, corridor_types::action> n_expected(0.3, 0.9, 1, corridor_pi);
        corridor_types::policy q_memory;
        corridor_types::policy e_memory;
        corridor_types::policy n_memory;
        for (const auto & e : corridor_episodes(200, 7)) {
            q_sigma(e.first, q_memory, e.second);
            expected(e.first, e_memory, e.second);
            n_expected(e.first, n_memory, e.second);
        }
        INFO("sigma = " << sigma);
        REQUIRE(largest_difference(q_memory, e_memory) < 1e-12);
        REQUIRE(largest_difference(n_memory, e_memory) < 1e-12);
    }
}

/// verifies an episode cut short, for the new learners: only the steps whose n rewards are known are learnt
TEST_CASE("n-step family: an episode cut short learns the steps whose n rewards are known", "[n_step]")
{
    // the worked episode, cut short after its second step: (s2, c) is the state reached and the action chosen next.
    // n = 2: only τ = 0 is learnt, and the same as when the episode goes on (G_{0:2} uses s2, not r2)
    auto cut = worked_episode();
    relearn::n_step_tree_backup<state,action> tree(0.5, 0.5, 2, worked_pi);
    relearn::n_step_q_sigma<state,action> q_sigma(0.5, 0.5, 2, 0.5, worked_pi);
    relearn::n_step_expected_sarsa<state,action> expected(0.5, 0.5, 2, worked_pi);
    auto tree_memory = worked_values();
    auto sigma_memory = worked_values();
    auto expected_memory = worked_values();
    tree(cut, tree_memory, false);
    q_sigma(cut, sigma_memory, false);
    expected(cut, expected_memory, false);
    REQUIRE(tree_memory.value(state("s0"), action("a")) == Approx(1.1875));
    REQUIRE(sigma_memory.value(state("s0"), action("a")) == Approx(1.40625));
    REQUIRE(expected_memory.value(state("s0"), action("a")) == Approx(1.375));
    // (s1, b) is not learnt, and (s2, c) keeps its value
    for (const auto * memory : {&tree_memory, &sigma_memory, &expected_memory}) {
        REQUIRE(memory->value(state("s1"), action("b")) == 0);
        REQUIRE(memory->value(state("s2"), action("c")) == Approx(4));
    }
}

/// verifies that learning online, after every step, learns exactly what learning after the episode does
TEST_CASE("n-step family: online with a window of n is the same as after the episode", "[n_step]")
{
    // each learner plays its own corridor with the same seeds, and the agent does not look at the values,
    // so all of them play the same episodes. π and b are fixed, so learning online updates the same steps,
    // in the same order, from the same values as learning after the episode. Half are cut short, after 4 moves
    using tree_type  = relearn::n_step_tree_backup<corridor_types::state, corridor_types::action>;
    using sigma_type = relearn::n_step_q_sigma<corridor_types::state, corridor_types::action>;
    using sarsa_type = relearn::off_policy_n_step_sarsa<corridor_types::state, corridor_types::action>;
    // play 100 episodes online with @param learner, and learn the same episodes after each one with @param after
    auto compare = [](auto online, auto after, unsigned int n) {
        corridor online_game(9);
        corridor after_game(9);
        std::mt19937 online_prng(10);
        std::mt19937 after_prng(10);
        std::bernoulli_distribution right(0.7);
        auto online_agent = [&](const int &, const std::vector<unsigned int> &) { return right(online_prng) ? 1u : 0u; };
        auto after_agent  = [&](const int &, const std::vector<unsigned int> &) { return right(after_prng) ? 1u : 0u; };
        corridor_types::policy online_memory;
        corridor_types::policy after_memory;
        for (int i = 0; i < 100; i++) {
            std::size_t limit = i % 2 == 0 ? 4 : 1000;
            games::play_online(online_game, online_agent, online, online_memory, limit, n);
            auto r = games::play(after_game, after_agent, limit);
            after(r.episode, after_memory, r.terminated);
        }
        return largest_difference(online_memory, after_memory);
    };
    for (unsigned int n : {1u, 3u}) {
        INFO("n = " << n);
        REQUIRE(compare(tree_type(0.3, 0.9, n, corridor_pi), tree_type(0.3, 0.9, n, corridor_pi), n) == 0);
        sigma_type q_sigma(0.3, 0.9, n, 0.5, corridor_pi);
        q_sigma.behaviour = corridor_b;
        REQUIRE(compare(q_sigma, q_sigma, n) == 0);
        sarsa_type sarsa(0.3, 0.9, n, corridor_pi, corridor_b);
        REQUIRE(compare(sarsa, sarsa, n) == 0);
    }
}

/// verifies the constructors, and brace initialisation
TEST_CASE("n-step family: constructors and brace initialisation", "[n_step]")
{
    relearn::n_step_expected_sarsa<state,action> expected;
    REQUIRE(expected.alpha == Approx(0.9));
    REQUIRE(expected.gamma == Approx(0.9));
    REQUIRE(expected.n == 1);
    REQUIRE(!expected.probabilities);
    REQUIRE(!expected.behaviour);
    relearn::n_step_q_sigma<state,action> q_sigma;
    REQUIRE(q_sigma.sigma == Approx(0.5));
    auto tree = relearn::n_step_tree_backup<state,action>{0.1, 1.0, 4, worked_pi};
    REQUIRE(tree.alpha == Approx(0.1));
    REQUIRE(tree.n == 4);
    REQUIRE(tree.probabilities);
    auto off = relearn::off_policy_n_step_sarsa<state,action>{0.2, 0.5, 3, worked_pi, worked_b};
    REQUIRE(off.gamma == Approx(0.5));
    REQUIRE(off.behaviour);
}

/// verifies on-policy control: each learns the exact values of the ε-greedy policy it plays and learns
TEST_CASE("n-step family: on-policy, learns q of the epsilon-greedy policy in the corridor", "[n_step][convergence]")
{
    // the agent plays ε-greedy (ε = 0.1) on the values it learns, and π is that same policy
    const double gamma = 0.9;
    auto exact = corridor::model(gamma);
    std::vector<std::vector<double>> pi(5, {0.05, 0.95});
    auto v_epsilon = exact.policy_values(pi);
    auto run = [&](auto learner, unsigned int n, unsigned int seed) {
        corridor game(seed);
        std::mt19937 prng(seed + 100);
        corridor_types::policy memory;
        learner.learning_rate = relearn::polynomial_decay<>(1.0, 0.6);
        learner.probabilities = [&memory](const corridor_types::state & s) {
            return games::epsilon_greedy_probabilities<corridor>(memory, s.trait(), {0u, 1u}, 0.1);
        };
        auto agent = [&](const int & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<corridor>(memory, s, legal, 0.1, prng);
        };
        for (int k = 1; k <= 20000; k++) {
            games::play_online(game, agent, learner, memory, 1000, n);
        }
        return worst_error(memory, exact, v_epsilon);
    };
    // over 10 seeds, with libstdc++ and with libc++, the worst errors were at most 0.097 with n = 2,
    // and 0.157 with n = 4 (n-step Expected SARSA 0.143, tree backup 0.107, Q(σ) 0.157 and 0.128)
    for (unsigned int n : {2u, 4u}) {
        INFO("n = " << n);
        REQUIRE(run(relearn::n_step_expected_sarsa<corridor_types::state, corridor_types::action>(0.1, gamma, n), n, 1) < 0.25);
        REQUIRE(run(relearn::n_step_tree_backup<corridor_types::state, corridor_types::action>(0.1, gamma, n), n, 1) < 0.25);
        REQUIRE(run(relearn::n_step_q_sigma<corridor_types::state, corridor_types::action>(0.1, gamma, n, 0.5), n, 1) < 0.25);
        REQUIRE(run(relearn::n_step_q_sigma<corridor_types::state, corridor_types::action>(0.1, gamma, n, 1.0), n, 1) < 0.25);
    }
}

/// verifies off-policy control: playing a fixed exploring policy, each learns the optimal values q*
TEST_CASE("n-step family: off-policy, learns q* in the corridor from another policy's play", "[n_step][convergence]")
{
    // the agent moves right with 0.7 and left with 0.3 whatever it has learnt; π is greedy on the values
    const double gamma = 0.9;
    auto exact = corridor::model(gamma);
    auto v_star = exact.optimal_values();
    auto greedy = [](auto & learner, const corridor_types::policy & memory) {
        learner.probabilities = [&memory](const corridor_types::state & s) {
            return games::epsilon_greedy_probabilities<corridor>(memory, s.trait(), {0u, 1u}, 0.0);
        };
    };
    auto greedy_and_b = [&greedy](auto & learner, const corridor_types::policy & memory) {
        greedy(learner, memory);
        learner.behaviour = corridor_b;
    };
    // over 10 seeds, the same with libstdc++ and with libc++, the worst errors were at most 0.045 with n = 2,
    // and 0.102 with n = 4 (tree backup 0.037, off-policy SARSA 0.051, Expected SARSA 0.102, Q(σ) 0.055)
    for (unsigned int n : {2u, 4u}) {
        INFO("n = " << n);
        auto tree = learn_corridor_with(relearn::n_step_tree_backup<corridor_types::state, corridor_types::action>(0.1, gamma, n),
                                        greedy, 0.7, 1.0, n, 1);
        REQUIRE(worst_error(tree, exact, v_star) < 0.25);
        // importance sampling: ρ reaches (1 / 0.7)^n, so its step sizes start at 0.2, keeping α ρ below 1
        auto sarsa = learn_corridor_with(relearn::off_policy_n_step_sarsa<corridor_types::state, corridor_types::action>(0.1, gamma, n),
                                         greedy_and_b, 0.7, 0.2, n, 1);
        REQUIRE(worst_error(sarsa, exact, v_star) < 0.25);
        auto expected = learn_corridor_with(relearn::n_step_expected_sarsa<corridor_types::state, corridor_types::action>(0.1, gamma, n),
                                            greedy_and_b, 0.7, 1.0, n, 1);
        REQUIRE(worst_error(expected, exact, v_star) < 0.25);
        auto q_sigma = learn_corridor_with(relearn::n_step_q_sigma<corridor_types::state, corridor_types::action>(0.1, gamma, n, 0.5),
                                           greedy_and_b, 0.7, 1.0, n, 1);
        REQUIRE(worst_error(q_sigma, exact, v_star) < 0.25);
    }
}
