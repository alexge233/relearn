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
