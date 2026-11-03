// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/game.hpp"
#include "../examples/games/random_walk.hpp"
#include "../examples/games/dyna_maze.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include "solvers.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of the eligibility traces (Sutton & Barto, Chapter 12): `sarsa_lambda`, `td_lambda`,
 * `watkins_q_lambda`, `true_online_sarsa_lambda` and `true_online_td_lambda`, `truncated_sarsa_lambda`
 * and `truncated_td_lambda`, `online_lambda_return`, and a λ and a γ that vary (`lambda_of`, `gamma_of`):
 * - traces and values worked out by hand, accumulating, replacing and Dutch, and Watkins's cut
 * - λ = 0 is `sarsa`, `q_learning` and TD(0), offline TD(λ) is the offline λ-return algorithm,
 *   and true online Sarsa(λ) is the online λ-return algorithm, after every step
 * - the 19-state random walk (Figure 12.6): an intermediate λ is best, and true online TD(λ)
 *   learns at least as well as TD(λ), and much better with a large α
 * - control: the exact values in a slippery corridor, and the Dyna maze after one episode, and early on
 * - truncated TD(λ) and the online λ-return algorithm worked out by hand and against their definitions, after every
 *   step; their equivalences (n = 1, λ = 1, a long n; λ = 0, λ = 1), and the 19-state walk
 * - a λ and a γ that vary: by hand, offline TD(λ) against the λ-return of Section 12.8, true online Sarsa(λ) against
 *   the online λ-return algorithm, γ = 0 splitting an episode, and γ = 0 in a state of the 19-state walk
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;

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
 * @brief the root mean square error of the learnt values of a random walk, at most 1
 * @param walk gives the true values
 * @param memory holds the learnt values, V(s) = Q(s, 0)
 * @return sqrt(Σ_s (V(s) - v(s))² / number of states), or 1 if larger (or if the values blew up)
 */
double rms_error(const walk_type & walk,
                 const walk_types::policy & memory)
{
    double sum = 0;
    for (unsigned int s = 1; s <= walk.states(); s++) {
        double error = memory.value(walk_types::state(s), walk_types::action(0)) - walk.true_value(s);
        sum += error * error;
    }
    double rms = std::sqrt(sum / walk.states());
    return (std::isfinite(rms) && rms < 1) ? rms : 1;
}

/**
 * @brief Figure 12.6: the RMS error of a TD(λ) learner on the 19-state walk, over its first 10 episodes
 * @param lambda is λ
 * @param seed seeds the walks of the runs (seed * 1000 + run)
 * @return for each α = 0.05, 0.1, ... 1, the error after each of the first 10 walks,
 *         averaged over them and over 50 runs
 */
template <class learner_type>
std::vector<double> figure_12_6(double lambda,
                                unsigned int seed)
{
    std::vector<double> errors;
    for (int i = 1; i <= 20; i++) {
        double error = 0;
        for (unsigned int run = 0; run < 50; run++) {
            walk_type walk(19, -1, 1, seed * 1000 + run);
            walk_types::policy memory;
            learner_type td(0.05 * i, 1.0, lambda);
            for (unsigned int k = 0; k < 10; k++) {
                auto r = games::play(walk, only_action, 100000);
                td(r.episode, memory, r.terminated);
                error += rms_error(walk, memory) / 500;
            }
        }
        errors.push_back(error);
    }
    return errors;
}

/**
 * @brief Figure 12.6: the lowest RMS error of TD(λ) on the 19-state walk, over its first 10 episodes
 * @param lambda is λ
 * @param seed seeds the walks of the runs (seed * 1000 + run)
 * @return the lowest, over α = 0.05, 0.1, ... 1, of the error after each of the first 10 walks,
 *         averaged over them and over 50 runs
 */
double best_of_figure_12_6(double lambda,
                           unsigned int seed)
{
    auto errors = figure_12_6<relearn::td_lambda<walk_types::state, walk_types::action>>(lambda, seed);
    return *std::min_element(errors.begin(), errors.end());
}

/**
 * @brief the λ-return algorithm offline (Section 12.1), written from its definition, for one episode
 * @param walk_episode is a finished episode of the random walk
 * @param memory holds the values, which are updated once the whole episode is worked out
 * @param alpha, @param gamma and @param lambda are α, γ and λ
 *
 * G_{t:t+n} = r_t + ... + γ^{n-1} r_{t+n-1} + γ^n V(s_{t+n}), and G_t the return to the end;
 * G^λ_t = (1 - λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_{t:t+n} + λ^{T-t-1} G_t, and every V(s_t) moves by
 * α (G^λ_t - V(s_t)), all of them worked out with the values from before the episode.
 */
void offline_lambda_return(const walk_types::episode & walk_episode,
                           walk_types::policy & memory,
                           double alpha,
                           double gamma,
                           double lambda)
{
    const auto before = memory;
    const std::size_t T = walk_episode.size();
    auto V = [&](std::size_t t) {
        return before.value(walk_episode[t].state, walk_episode[t].action);
    };
    for (std::size_t t = 0; t < T; t++) {
        // the n-step returns, n = 1 ... T-t, the last one being the return to the end
        std::vector<double> G;
        double rewards = 0;
        double discount = 1;
        for (std::size_t k = t; k < T; k++) {
            rewards += discount * walk_episode[k].state.reward();
            discount *= gamma;
            G.push_back(k + 1 < T ? rewards + discount * V(k + 1) : rewards);
        }
        // G^λ_t: the n-step returns weighted by (1 - λ) λ^{n-1}, and the rest of the weight on G_t
        double lambda_return = 0;
        double weight = 1;
        for (std::size_t n = 1; n < G.size(); n++) {
            lambda_return += (1 - lambda) * weight * G[n - 1];
            weight *= lambda;
        }
        lambda_return += weight * G.back();
        // V(s_t) moves by α (G^λ_t - V(s_t)), the moves of every visit adding up
        const auto & s = walk_episode[t];
        memory.update(s.state, s.action, memory.value(s.state, s.action) + alpha * (lambda_return - V(t)));
    }
}

/**
 * @brief the online λ-return algorithm (Section 12.4), written from its definition, for one episode
 * @param e is the episode: steps with integer states and actions, and their rewards
 * @param start holds the values at the start of the episode
 * @param alpha, @param gamma and @param lambda are α, γ and λ
 * @param terminated is false for an episode cut short, whose last link is not a step
 * @return the values after every step h = 0, 1, ..., T of the episode: element h is w_h
 *
 * After step h, every update of the episode is worked out again from its start, from the values `start`,
 * each with the λ-return truncated at h:
 * G^λ_{t:h} = (1 - λ) Σ_{n=1}^{h-t-1} λ^{n-1} G_{t:t+n} + λ^{h-t-1} G_{t:h}, where
 * G_{t:t+n} = r_t + ... + γ^{n-1} r_{t+n-1} + γ^n Q(s_{t+n}, a_{t+n}), the value being the one at step t+n-1,
 * w_{t+n-1} (none after the end of the episode). The values at h, w_h, are those of the last update.
 */
std::vector<policy> online_lambda_return(const episode & e,
                                         const policy & start,
                                         double alpha,
                                         double gamma,
                                         double lambda,
                                         bool terminated)
{
    const std::size_t T = terminated ? e.size() : e.size() - 1;
    std::vector<policy> w {start};
    for (std::size_t h = 1; h <= T; h++) {
        policy redone = start;
        for (std::size_t t = 0; t < h; t++) {
            // the n-step return G_{t:t+n}, bootstrapping from the values at step t+n-1
            auto n_step = [&](std::size_t n) {
                double G = 0;
                double discount = 1;
                for (std::size_t k = t; k < t + n; k++) {
                    G += discount * e[k].state.reward();
                    discount *= gamma;
                }
                if (t + n < e.size()) {
                    G += discount * w[t + n - 1].value(e[t + n].state, e[t + n].action);
                }
                return G;
            };
            // G^λ_{t:h}: the n-step returns weighted by (1 - λ) λ^{n-1}, and the rest of the weight on G_{t:h}
            double G = 0;
            double weight = 1;
            for (std::size_t n = 1; t + n < h; n++) {
                G += (1 - lambda) * weight * n_step(n);
                weight *= lambda;
            }
            G += weight * n_step(h - t);
            // Q(s_t, a_t) moves by α (G^λ_{t:h} - Q(s_t, a_t))
            double q = redone.value(e[t].state, e[t].action);
            redone.update(e[t].state, e[t].action, q + alpha * (G - q));
        }
        w.push_back(redone);
    }
    return w;
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
 * @brief learn the corridor, one whole episode at a time, or after every step
 * @param learner is the learner to train
 * @param seed seeds the corridor and the agent
 * @param random_play is true for an agent that moves at random, false for ε-greedy on what it learns (ε = 0.1)
 * @param online is true to learn after every step (`games::play_online`), false after each episode
 * @return the learnt policy, after 20000 episodes, with α = 0.1 * 1000 / (1000 + k) in episode k
 */
template <class learner_type>
corridor_types::policy learn_corridor(learner_type & learner,
                                      unsigned int seed,
                                      bool random_play,
                                      bool online = false)
{
    corridor game(seed);
    std::mt19937 prng(seed + 100);
    corridor_types::policy memory;
    auto agent = [&](const int & s,
                     const std::vector<unsigned int> & legal) {
        if (random_play) {
            std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
            return legal[pick(prng)];
        }
        return games::epsilon_greedy<corridor>(memory, s, legal, 0.1, prng);
    };
    for (int k = 1; k <= 20000; k++) {
        learner.alpha = 0.1 * 1000.0 / (1000.0 + k);
        if (online) {
            games::play_online(game, agent, learner, memory, 1000);
        }
        else {
            auto r = games::play(game, agent, 1000);
            learner(r.episode, memory, r.terminated);
        }
    }
    return memory;
}

/**
 * @brief the mean moves of episodes 2 to 10 in the Dyna maze, over 30 runs
 * @param learner is the learner, copied afresh for every run
 * @param seed seeds the runs (seed * 1000 + run)
 * @return the mean moves (the first episode is a random walk whatever λ is, so it is left out)
 *
 * ε-greedy with ε = 0.1, and each episode learnt once it has ended
 */
template <class learner_type>
double early_moves(const learner_type & learner,
                   unsigned int seed)
{
    double moves = 0;
    for (unsigned int run = 0; run < 30; run++) {
        std::mt19937 prng(seed * 1000 + run);
        maze_type maze;
        maze_types::policy memory;
        learner_type fresh = learner;
        auto agent = [&](const maze_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<maze_type>(memory, s, legal, 0.1, prng);
        };
        for (unsigned int i = 1; i <= 10; i++) {
            auto r = games::play(maze, agent, 100000);
            fresh(r.episode, memory, r.terminated);
            if (i > 1) {
                moves += static_cast<double>(r.steps) / (9 * 30);
            }
        }
    }
    return moves;
}

/**
 * @brief play the first episode of the Dyna maze, learn it, and follow the greedy policy learnt
 * @param learner is the learner
 * @param seed seeds the agent
 * @param first receives the first episode
 * @param memory receives what was learnt from it
 * @return the greedy episode, at most 100 moves
 */
template <class learner_type>
games::played<maze_type> after_one_episode(learner_type & learner,
                                           unsigned int seed,
                                           games::played<maze_type> & first,
                                           maze_types::policy & memory)
{
    std::mt19937 prng(seed);
    maze_type maze;
    auto agent = [&](const maze_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<maze_type>(memory, s, legal, 0.1, prng);
    };
    first = games::play(maze, agent, 100000);
    learner(first.episode, memory, first.terminated);
    auto greedy = [&](const maze_type::state_type & s,
                      const std::vector<unsigned int> & legal) {
        return games::greedy<maze_type>(memory, s, legal, prng);
    };
    return games::play(maze, greedy, 100);
}
} // namespace

/// verifies SARSA(λ)'s traces and values worked out by hand, with accumulating and with replacing traces
TEST_CASE("sarsa_lambda: traces and values worked out by hand", "[eligibility_traces][sarsa_lambda]")
{
    // γ = 0.5, λ = 0.5 (so γλ = 0.25), α = 0.5: (x, a) earns 1, (y, b) 0, then (x, a) again 2, and the episode ends
    //    t = 0: δ = 1 + 0.5 Q(y, b) - Q(x, a) = 1,          z(x, a) = 1,     Q(x, a) = 0.5
    //    t = 1: δ = 0 + 0.5 Q(x, a) - Q(y, b) = 0.25,       z(x, a) = 0.25, z(y, b) = 1
    //           Q(x, a) = 0.5 + 0.5 * 0.25 * 0.25 = 0.53125, Q(y, b) = 0.5 * 0.25 * 1 = 0.125
    //    t = 2: δ = 2 - Q(x, a) = 1.46875,                  z(y, b) = 0.25, and z(x, a) = 0.0625, then
    //           accumulating: z(x, a) = 1.0625, Q(x, a) = 0.53125 + 0.5 * 1.46875 * 1.0625 = 1.3115234375
    //           replacing:    z(x, a) = 1,      Q(x, a) = 0.53125 + 0.5 * 1.46875 * 1      = 1.265625
    //           both:         Q(y, b) = 0.125 + 0.5 * 1.46875 * 0.25 = 0.30859375
    auto twice = episode {{state(1, "x"), action("a")},
                          {state(0, "y"), action("b")},
                          {state(2, "x"), action("a")}};
    SECTION("accumulating") {
        relearn::sarsa_lambda<state,action> learner(0.5, 0.5, 0.5);
        policy memory;
        learner(twice, memory);
        REQUIRE(memory.value(state("x"), action("a")) == Approx(1.3115234375));
        REQUIRE(memory.value(state("y"), action("b")) == Approx(0.30859375));
        REQUIRE(learner.traces.trace(state("x"), action("a")) == Approx(1.0625));
        REQUIRE(learner.traces.trace(state("y"), action("b")) == Approx(0.25));
        REQUIRE(learner.traces.entries().size() == 2);
    }
    SECTION("replacing") {
        relearn::sarsa_lambda<state,action> learner(0.5, 0.5, 0.5, true);
        policy memory;
        learner(twice, memory);
        REQUIRE(memory.value(state("x"), action("a")) == Approx(1.265625));
        REQUIRE(memory.value(state("y"), action("b")) == Approx(0.30859375));
        REQUIRE(learner.traces.trace(state("x"), action("a")) == Approx(1));
        REQUIRE(learner.traces.trace(state("y"), action("b")) == Approx(0.25));
    }
}

/// verifies Watkins's Q(λ) worked out by hand: the traces are cut after a move that is not greedy
TEST_CASE("watkins_q_lambda: the traces are cut after a move that is not greedy", "[eligibility_traces][q_lambda]")
{
    // γ = 0.5, λ = 1 (so γλ = 0.5), α = 0.5, and Q(y, c) = 4 before the episode: c is the greedy move in y.
    // The episode moves (x, a), then (y, b), which is not greedy, then (z, d), which earns 1 and ends it
    auto explored = episode {{state(0, "x"), action("a")},
                             {state(0, "y"), action("b")},
                             {state(1, "z"), action("d")}};
    SECTION("watkins_q_lambda") {
        //    t = 0: δ = 0 + 0.5 max Q(y, .) - Q(x, a) = 0.5 * 4 = 2,  z(x, a) = 1,  Q(x, a) = 0.5 * 2 * 1 = 1
        //           b is not greedy in y (0 < 4): the traces are cut before the next step
        //    t = 1: z(x, a) = 0; δ = 0 + 0.5 * 0 - Q(y, b) = 0,         z(y, b) = 1
        //    t = 2: z(y, b) = 0.5; δ = 1 - Q(z, d) = 1,                  z(z, d) = 1
        //           Q(z, d) = 0.5, Q(y, b) = 0.5 * 1 * 0.5 = 0.25, and Q(x, a) stays 1
        relearn::watkins_q_lambda<state,action> learner(0.5, 0.5, 1.0);
        policy memory;
        memory.update(state("y"), action("c"), 4);
        learner(explored, memory);
        REQUIRE(memory.value(state("x"), action("a")) == Approx(1));
        REQUIRE(memory.value(state("y"), action("b")) == Approx(0.25));
        REQUIRE(memory.value(state("z"), action("d")) == Approx(0.5));
        REQUIRE(learner.traces.trace(state("x"), action("a")) == Approx(0));
        REQUIRE(learner.traces.trace(state("y"), action("b")) == Approx(0.5));
        REQUIRE(learner.traces.trace(state("z"), action("d")) == Approx(1));
    }
    SECTION("sarsa_lambda, on the same episode, is not cut") {
        //    t = 0: δ = 0 + 0.5 Q(y, b) - Q(x, a) = 0,  t = 1: δ = 0 + 0.5 Q(z, d) - Q(y, b) = 0
        //    t = 2: z(x, a) = 0.25, z(y, b) = 0.5, z(z, d) = 1, δ = 1:
        //           Q(x, a) = 0.125, Q(y, b) = 0.25, Q(z, d) = 0.5
        relearn::sarsa_lambda<state,action> learner(0.5, 0.5, 1.0);
        policy memory;
        memory.update(state("y"), action("c"), 4);
        learner(explored, memory);
        REQUIRE(memory.value(state("x"), action("a")) == Approx(0.125));
        REQUIRE(memory.value(state("y"), action("b")) == Approx(0.25));
        REQUIRE(memory.value(state("z"), action("d")) == Approx(0.5));
        REQUIRE(learner.traces.trace(state("x"), action("a")) == Approx(0.25));
    }
}

/// verifies an episode cut short: its last link is not a step, and only gives the next state and action
/// verifies that Watkins's Q(λ) decides whether a' is greedy with the values from before the step's update
TEST_CASE("watkins_q_lambda: a' is greedy or not before the update", "[eligibility_traces][watkins]")
{
    // α = 1, γ = 1, λ = 1, accumulating, with Q(x, a) = 0 and Q(x, b) = 1 to start with.
    // The episode is (x, a) earning 1, then (x, b) earning 3, which ends it: both steps are in x.
    // t = 0: a' = b, and before the update b is the greedy action of x (1 > 0), so the traces are kept.
    //        δ = 1 + 1 * max Q(x, .) - Q(x, a) = 1 + 1 - 0 = 2, and Q(x, a) = 0 + 2 = 2.
    //        (after the update, a would be the greedy action of x, and the traces would be cut)
    // t = 1: z(x, a) = γλ * 1 = 1; δ = 3 - Q(x, b) = 3 - 1 = 2, z(x, b) = 1, so
    //        Q(x, a) = 2 + 2 * 1 = 4 (through its trace) and Q(x, b) = 1 + 2 = 3.
    //        Had the traces been cut, Q(x, a) would have stayed 2
    relearn::watkins_q_lambda<state,action> learner(1.0, 1.0, 1.0);
    policy memory;
    memory.update(state("x"), action("a"), 0);
    memory.update(state("x"), action("b"), 1);
    learner(episode {{state(1, "x"), action("a")},
                     {state(3, "x"), action("b")}}, memory);
    REQUIRE(memory.value(state("x"), action("a")) == Approx(4));
    REQUIRE(memory.value(state("x"), action("b")) == Approx(3));
}

/// verifies that a trace which falls to exactly 0 is dropped, so that the traces do not pile up
TEST_CASE("eligibility_traces: traces at 0 are dropped", "[eligibility_traces][regression]")
{
    // a corridor of 100 different pairs, which ends with a reward of 1
    episode walk;
    for (int i = 0; i < 100; i++) {
        walk.push_back({state(i == 99 ? 1 : 0, "s" + std::to_string(i)), action("right")});
    }
    // λ = 0: every trace is cut to 0 before the next visit, so only the last pair is left,
    // and the values are one-step SARSA's (the test above checks them)
    relearn::sarsa_lambda<state,action> sarsa(0.5, 0.9, 0.0);
    policy memory;
    sarsa(walk, memory);
    REQUIRE(sarsa.traces.entries().size() == 1);
    REQUIRE(sarsa.traces.trace(state("s99"), action("right")) == Approx(1));
    REQUIRE(memory.value(state("s99"), action("right")) == Approx(0.5));
    // Watkins's Q(λ) with λ = 0.9: a non-greedy move cuts every trace, and only the pairs after it are left.
    // In s1, x is worth 1 and is the greedy action, so the move "right" in s1 is not greedy
    relearn::watkins_q_lambda<state,action> q(0.5, 0.9, 0.9);
    policy q_memory;
    q_memory.update(state("s1"), action("x"), 1);
    q(episode {{state(0, "s0"), action("right")},
               {state(0, "s1"), action("right")},
               {state(0, "s2"), action("right")},
               {state(1, "s3"), action("right")}}, q_memory);
    // the cut comes after the step from s0, so s0's trace is gone, and s1, s2 and s3 have theirs
    REQUIRE(q.traces.entries().size() == 3);
    REQUIRE(q.traces.trace(state("s0"), action("right")) == Approx(0));
}

TEST_CASE("sarsa_lambda and watkins_q_lambda: an episode cut short", "[eligibility_traces]")
{
    // γ = 1, λ = 1, α = 1: (s0, a) earns 1, then the state reached, (s1, b), worth 3 (its reward is not a reward)
    //    δ = 1 + Q(s1, b) - Q(s0, a) = 4, so Q(s0, a) = 4, and (s1, b) has no trace
    auto cut = episode {{state(1, "s0"), action("a")},
                        {state(100, "s1"), action("b")}};
    relearn::sarsa_lambda<state,action> sarsa(1.0, 1.0, 1.0);
    relearn::watkins_q_lambda<state,action> q(1.0, 1.0, 1.0);
    policy sarsa_memory;
    policy q_memory;
    sarsa_memory.update(state("s1"), action("b"), 3);
    q_memory.update(state("s1"), action("b"), 3);
    sarsa(cut, sarsa_memory, false);
    q(cut, q_memory, false);
    REQUIRE(sarsa_memory.value(state("s0"), action("a")) == Approx(4));
    REQUIRE(sarsa_memory.value(state("s1"), action("b")) == Approx(3));
    REQUIRE(sarsa.traces.entries().size() == 1);
    REQUIRE(q_memory.value(state("s0"), action("a")) == Approx(4));
    REQUIRE(q_memory.value(state("s1"), action("b")) == Approx(3));
    REQUIRE(q.traces.entries().size() == 1);
    // each episode starts its traces again: a second call does not carry the first one's
    sarsa(episode {{state(0, "s2"), action("c")}}, sarsa_memory);
    REQUIRE(sarsa.traces.entries().size() == 1);
    REQUIRE(sarsa.traces.trace(state("s0"), action("a")) == Approx(0));
}

/// verifies that with λ = 0, SARSA(λ) is SARSA, Q(λ) is Q-learning, and TD(λ) is TD(0)
TEST_CASE("eligibility traces: lambda = 0 is sarsa, q_learning and TD(0)", "[eligibility_traces]")
{
    // the same corridor episodes, played at random, some of them cut short after 5 moves.
    // The values agree to the last few bits only: α δ z is rounded before it is added, where
    // the one-step learners' q + α δ may be worked out in one go (a fused multiply-add)
    corridor game(3);
    std::mt19937 prng(4);
    auto agent = [&](const int &,
                     const std::vector<unsigned int> & legal) {
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        return legal[pick(prng)];
    };
    using S = corridor_types::state;
    using A = corridor_types::action;
    relearn::sarsa<S,A> one_step(0.3, 0.9);
    relearn::sarsa_lambda<S,A> sarsa_zero(0.3, 0.9, 0.0);
    relearn::q_learning<S,A> q_learning(0.3, 0.9);
    relearn::watkins_q_lambda<S,A> q_zero(0.3, 0.9, 0.0);
    corridor_types::policy one_memory, sarsa_memory, q_memory, q_zero_memory;
    for (int i = 0; i < 200; i++) {
        auto r = games::play(game, agent, i % 2 == 0 ? 5 : 1000);
        one_step(r.episode, one_memory, r.terminated);
        sarsa_zero(r.episode, sarsa_memory, r.terminated);
        q_learning(r.episode, q_memory, r.terminated);
        q_zero(r.episode, q_zero_memory, r.terminated);
    }
    for (int s = 0; s < 4; s++) {
        for (unsigned int a = 0; a < 2; a++) {
            REQUIRE(sarsa_memory.value(S(s), A(a)) == Approx(one_memory.value(S(s), A(a))).epsilon(1e-12));
            REQUIRE(q_zero_memory.value(S(s), A(a)) == Approx(q_memory.value(S(s), A(a))).epsilon(1e-12));
        }
    }
    // TD(λ = 0) on the random walk is n-step TD with n = 1
    walk_type walk_a(19, -1, 1, 6);
    walk_type walk_b(19, -1, 1, 6);
    walk_types::policy td_memory, zero_memory;
    relearn::n_step_td<walk_types::state, walk_types::action> td(0.4, 1.0, 1);
    relearn::td_lambda<walk_types::state, walk_types::action> td_zero(0.4, 1.0, 0.0);
    for (int i = 0; i < 50; i++) {
        auto r = games::play(walk_a, only_action, 100000);
        td(r.episode, td_memory, r.terminated);
        auto q = games::play(walk_b, only_action, 100000);
        td_zero(q.episode, zero_memory, q.terminated);
    }
    for (unsigned int s = 1; s <= 19; s++) {
        REQUIRE(zero_memory.value(walk_types::state(s), walk_types::action(0)) ==
                Approx(td_memory.value(walk_types::state(s), walk_types::action(0))).epsilon(1e-12).margin(1e-15));
    }
}

/// verifies that offline TD(λ), with accumulating traces, makes exactly the changes of the λ-return algorithm,
/// added up at the end of each episode (Section 12.1, and the first edition's Section 7.4)
TEST_CASE("td_lambda: offline, it is the offline lambda-return algorithm", "[eligibility_traces][td_lambda]")
{
    // random walks of the 19-state walk, which visit states many times, from values that are not 0
    for (double gamma : {1.0, 0.8}) {
        for (double lambda : {0.0, 0.5, 0.9, 1.0}) {
            walk_type walk(19, -1, 1, 7);
            walk_types::policy traced;
            walk_types::policy forward;
            for (unsigned int s = 1; s <= 19; s++) {
                traced.update(walk_types::state(s), walk_types::action(0), 0.05 * s - 0.5);
                forward.update(walk_types::state(s), walk_types::action(0), 0.05 * s - 0.5);
            }
            relearn::td_lambda<walk_types::state, walk_types::action> td(0.1, gamma, lambda);
            td.offline = true;
            for (int i = 0; i < 20; i++) {
                auto r = games::play(walk, only_action, 100000);
                td(r.episode, traced, r.terminated);
                offline_lambda_return(r.episode, forward, 0.1, gamma, lambda);
            }
            for (unsigned int s = 1; s <= 19; s++) {
                INFO("γ = " << gamma << ", λ = " << lambda << ", state " << s);
                REQUIRE(traced.value(walk_types::state(s), walk_types::action(0)) ==
                        Approx(forward.value(walk_types::state(s), walk_types::action(0))).epsilon(1e-9).margin(1e-12));
            }
        }
    }
}

/// verifies the constructors, and brace initialisation
TEST_CASE("eligibility traces: constructors and brace initialisation", "[eligibility_traces]")
{
    relearn::sarsa_lambda<state,action> defaults;
    REQUIRE(defaults.alpha == Approx(0.9));
    REQUIRE(defaults.gamma == Approx(0.9));
    REQUIRE(defaults.lambda == Approx(0.9));
    REQUIRE_FALSE(defaults.replacing);
    REQUIRE_FALSE(defaults.offline);
    auto sarsa = relearn::sarsa_lambda<state,action>{0.1, 1.0, 0.8, true};
    REQUIRE(sarsa.alpha == Approx(0.1));
    REQUIRE(sarsa.gamma == Approx(1));
    REQUIRE(sarsa.lambda == Approx(0.8));
    REQUIRE(sarsa.replacing);
    auto q = relearn::watkins_q_lambda<state,action>{0.2, 0.95, 0.7};
    REQUIRE(q.alpha == Approx(0.2));
    REQUIRE(q.gamma == Approx(0.95));
    REQUIRE(q.lambda == Approx(0.7));
    REQUIRE_FALSE(q.replacing);
    relearn::td_lambda<state,action> td(0.3, 0.5, 0.4);
    REQUIRE(td.lambda == Approx(0.4));
}

/// verifies Figure 12.6: on the 19-state walk, an intermediate λ learns best in the first 10 episodes
TEST_CASE("td_lambda: an intermediate lambda is best (Figure 12.6)", "[eligibility_traces][td_lambda][random_walk]")
{
    // each λ at its best α (0.05 ... 1), over 50 runs. Over 10 seeds, with libstdc++ and with libc++,
    // the errors were λ = 0: 0.341 to 0.348, λ = 0.4: 0.266 to 0.279, λ = 0.8: 0.253 to 0.270,
    // λ = 0.9: 0.280 to 0.297, and λ = 1: 0.58 to 0.64 (worse than not learning at all, 0.55, for every α tried).
    // λ = 0.8 was the best every time, by at least 0.008
    std::vector<double> lambdas = {0.0, 0.4, 0.8, 0.9, 1.0};
    std::vector<double> best;
    for (auto lambda : lambdas) {
        best.push_back(best_of_figure_12_6(lambda, 1));
    }
    INFO("λ = 0: " << best[0] << ", 0.4: " << best[1] << ", 0.8: " << best[2]
         << ", 0.9: " << best[3] << ", 1: " << best[4]);
    // λ = 0.8 is the best, with about 0.25; λ = 0 and λ = 1 are much worse
    REQUIRE(best[2] < best[1]);
    REQUIRE(best[2] < best[3]);
    REQUIRE(best[2] < best[0] - 0.05);
    REQUIRE(best[2] < best[4] - 0.05);
    REQUIRE(best[2] == Approx(0.25).margin(0.03));
}

/// verifies SARSA(λ) control: with a constant ε it learns the exact values of the ε-greedy policy
TEST_CASE("sarsa_lambda: learns q of the epsilon-greedy policy in the corridor", "[eligibility_traces][sarsa_lambda][convergence]")
{
    // the best move is right in every cell, so the ε-greedy policy (ε = 0.1) moves right with 0.95, left 0.05
    auto exact = corridor::model(0.9);
    std::vector<std::vector<double>> pi(5, {0.05, 0.95});
    auto v_epsilon = exact.policy_values(pi);
    // over 10 seeds, with libstdc++ and with libc++, the worst errors were
    // 0.012 to 0.121, for λ = 0.5 and 0.9, accumulating or replacing
    for (double lambda : {0.5, 0.9}) {
        for (bool replacing : {false, true}) {
            relearn::sarsa_lambda<corridor_types::state, corridor_types::action> learner(0.1, 0.9, lambda, replacing);
            auto memory = learn_corridor(learner, 1, false);
            INFO("λ = " << lambda << (replacing ? ", replacing" : ", accumulating"));
            REQUIRE(worst_error(memory, exact, v_epsilon) < 0.2);
        }
    }
}

/// verifies Watkins's Q(λ) control: off-policy, it learns q*, from ε-greedy play and even from random play
TEST_CASE("watkins_q_lambda: learns q* in the corridor", "[eligibility_traces][q_lambda][convergence]")
{
    auto exact  = corridor::model(0.9);
    auto v_star = exact.optimal_values();
    // over 10 seeds, with libstdc++ and with libc++, the worst errors were 0.014 to 0.072
    for (double lambda : {0.5, 0.9}) {
        for (bool random_play : {false, true}) {
            relearn::watkins_q_lambda<corridor_types::state, corridor_types::action> learner(0.1, 0.9, lambda);
            auto memory = learn_corridor(learner, 1, random_play);
            INFO("λ = " << lambda << (random_play ? ", random play" : ", ε-greedy"));
            REQUIRE(worst_error(memory, exact, v_star) < 0.15);
        }
    }
}

/// verifies that with replacing traces, one episode is enough to find the goal again, without a loop
TEST_CASE("eligibility traces: after one episode of the maze, the greedy path has no loop", "[eligibility_traces][dyna_maze]")
{
    // every value is 0 until the goal is reached, so the only δ is the last one, and each move learns
    // α (γλ)^k, k being the steps since it was last made: in every cell, the move made last is the best.
    // Following them, each cell is left later than the one before, so the path cannot loop, and it
    // reaches the goal in at most as many moves as there are cells visited
    for (double lambda : {0.5, 0.9}) {
        for (unsigned int seed : {1u, 2u, 3u}) {
            relearn::sarsa_lambda<maze_types::state, maze_types::action> sarsa(0.1, 0.95, lambda, true);
            relearn::watkins_q_lambda<maze_types::state, maze_types::action> q(0.1, 0.95, lambda, true);
            games::played<maze_type> first_sarsa, first_q;
            maze_types::policy sarsa_memory, q_memory;
            auto greedy_sarsa = after_one_episode(sarsa, seed, first_sarsa, sarsa_memory);
            auto greedy_q     = after_one_episode(q, seed, first_q, q_memory);
            std::set<std::pair<int,int>> cells;
            for (const auto & link : first_sarsa.episode) {
                cells.insert({link.state.trait().x, link.state.trait().y});
            }
            INFO("λ = " << lambda << ", seed " << seed << ": first episode " << first_sarsa.steps << " moves");
            REQUIRE(greedy_sarsa.terminated);
            REQUIRE(greedy_sarsa.steps <= cells.size());
            REQUIRE(greedy_q.terminated);
            REQUIRE(greedy_q.steps == greedy_sarsa.steps);
        }
    }
}

/// verifies that traces learn the maze faster than one-step methods, and replacing traces faster than accumulating ones
TEST_CASE("eligibility traces: the maze is learnt faster with traces", "[eligibility_traces][dyna_maze]")
{
    // the mean moves of episodes 2 to 10, over 30 runs, α = 0.1, γ = 0.95. Over 10 seeds, with
    // libstdc++ and with libc++, they were: λ = 0, SARSA 277 to 350, Q-learning 263 to 326;
    // λ = 0.9 with replacing traces, 20.6 to 23.0 (both); λ = 0.9 with accumulating traces,
    // SARSA(λ) 82 to 132, Q(λ) 32 to 83
    using S = maze_types::state;
    using A = maze_types::action;
    double sarsa_zero  = early_moves(relearn::sarsa_lambda<S,A>(0.1, 0.95, 0.0), 1);
    double sarsa_accum = early_moves(relearn::sarsa_lambda<S,A>(0.1, 0.95, 0.9), 1);
    double sarsa_repl  = early_moves(relearn::sarsa_lambda<S,A>(0.1, 0.95, 0.9, true), 1);
    double q_zero      = early_moves(relearn::watkins_q_lambda<S,A>(0.1, 0.95, 0.0), 1);
    double q_repl      = early_moves(relearn::watkins_q_lambda<S,A>(0.1, 0.95, 0.9, true), 1);
    INFO("SARSA(0) " << sarsa_zero << ", SARSA(0.9) accumulating " << sarsa_accum << ", replacing " << sarsa_repl
         << ", Q(0) " << q_zero << ", Q(0.9) replacing " << q_repl);
    REQUIRE(sarsa_repl < 0.2 * sarsa_zero);
    REQUIRE(q_repl < 0.2 * q_zero);
    REQUIRE(sarsa_repl < 0.5 * sarsa_accum);
}

/// verifies true online Sarsa(λ)'s Dutch traces and values worked out by hand
TEST_CASE("true_online_sarsa_lambda: traces and values worked out by hand", "[eligibility_traces][true_online]")
{
    // the episode of the SARSA(λ) test above: γ = 0.5, λ = 0.5 (so γλ = 0.25, αγλ = 0.125), α = 0.5,
    // (x, a) earns 1, (y, b) 0, then (x, a) again 2, and the episode ends. Q and Q' are read before each step
    //    t = 0: Q = 0, Q' = Q(y, b) = 0, δ = 1,               z(x, a) = 1 - 0.125 * 0 = 1
    //           Q(x, a) = 0 + 0.5 * (1 + 0 - 0) * 1 = 0.5, minus 0.5 * (0 - 0);  Q_old = 0
    //    t = 1: Q = 0, Q' = Q(x, a) = 0.5, δ = 0 + 0.25 = 0.25, z(x, a) = 0.25, z(y, b) = 1
    //           Q(x, a) = 0.5 + 0.125 * 0.25 = 0.53125, Q(y, b) = 0.125;  Q_old = 0.5
    //    t = 2: Q = 0.53125, Q' = 0 (the end), δ = 2 - 0.53125 = 1.46875
    //           z(y, b) = 0.25, z(x, a) = 0.25 * 0.25 + 1 - 0.125 * 0.25 = 1.03125 (between replacing, 1, and accumulating, 1.0625)
    //           α (δ + Q - Q_old) = 0.5 * (1.46875 + 0.53125 - 0.5) = 0.75, so
    //           Q(x, a) = 0.53125 + 0.75 * 1.03125 - 0.5 * (0.53125 - 0.5) = 1.2890625, Q(y, b) = 0.125 + 0.75 * 0.25 = 0.3125
    auto twice = episode {{state(1, "x"), action("a")},
                          {state(0, "y"), action("b")},
                          {state(2, "x"), action("a")}};
    relearn::true_online_sarsa_lambda<state,action> learner(0.5, 0.5, 0.5);
    policy memory;
    learner(twice, memory);
    REQUIRE(memory.value(state("x"), action("a")) == Approx(1.2890625));
    REQUIRE(memory.value(state("y"), action("b")) == Approx(0.3125));
    REQUIRE(learner.traces.trace(state("x"), action("a")) == Approx(1.03125));
    REQUIRE(learner.traces.trace(state("y"), action("b")) == Approx(0.25));
    REQUIRE(learner.traces.entries().size() == 2);
    REQUIRE(learner.q_old == Approx(0));
}

/// verifies the exact equivalence: after every step, true online Sarsa(λ) has the values of the online λ-return algorithm
TEST_CASE("true_online_sarsa_lambda: after every step, it is the online lambda-return algorithm", "[eligibility_traces][true_online]")
{
    // random episodes of 4 states and 2 actions, which come back to the same pairs, from random values,
    // with random α, γ and λ (γ = 1 and λ = 1 too), and a third of them cut short
    std::mt19937 prng(5);
    std::uniform_int_distribution<int> pick_state(0, 3);
    std::uniform_int_distribution<int> pick_action(0, 1);
    std::uniform_int_distribution<int> pick_reward(-2, 3);
    std::uniform_int_distribution<std::size_t> pick_length(1, 12);
    std::uniform_real_distribution<double> uniform(0, 1);
    double largest = 0;
    for (int trial = 0; trial < 200; trial++) {
        const double alpha  = 0.05 + 0.9 * uniform(prng);
        const double gamma  = trial % 5 == 0 ? 1.0 : uniform(prng);
        const double lambda = trial % 7 == 0 ? 1.0 : uniform(prng);
        const bool terminated = trial % 3 != 0;
        episode e;
        const std::size_t length = pick_length(prng);
        for (std::size_t i = 0; i < length; i++) {
            e.push_back({state(pick_reward(prng), "s" + std::to_string(pick_state(prng))),
                         action("a" + std::to_string(pick_action(prng)))});
        }
        policy start;
        for (int s = 0; s < 4; s++) {
            for (int a = 0; a < 2; a++) {
                start.update(state("s" + std::to_string(s)), action("a" + std::to_string(a)), uniform(prng) - 0.5);
            }
        }
        auto expected = online_lambda_return(e, start, alpha, gamma, lambda, terminated);
        // learnt after every step, as `games::play_online` gives it: (s_t, a_t) and the next link, then the last step alone
        relearn::true_online_sarsa_lambda<state,action> online(alpha, gamma, lambda);
        online.start_episode();
        policy memory = start;
        const std::size_t T = terminated ? e.size() : e.size() - 1;
        for (std::size_t t = 0; t < T; t++) {
            if (t + 1 < e.size()) {
                online(episode {e[t], e[t + 1]}, memory, false);
            }
            else {
                online(episode {e[t]}, memory, true);
            }
            for (int s = 0; s < 4; s++) {
                for (int a = 0; a < 2; a++) {
                    state s_k("s" + std::to_string(s));
                    action a_k("a" + std::to_string(a));
                    largest = std::max(largest, std::fabs(memory.value(s_k, a_k) - expected[t + 1].value(s_k, a_k)));
                }
            }
        }
        // and learnt in one call, it is the very same, to the last bit
        relearn::true_online_sarsa_lambda<state,action> whole(alpha, gamma, lambda);
        policy at_once = start;
        whole(e, at_once, terminated);
        for (int s = 0; s < 4; s++) {
            for (int a = 0; a < 2; a++) {
                state s_k("s" + std::to_string(s));
                action a_k("a" + std::to_string(a));
                REQUIRE(at_once.value(s_k, a_k) == memory.value(s_k, a_k));
            }
        }
    }
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 1e-12);
}

/// verifies that with λ = 0, true online Sarsa(λ) is SARSA, and true online TD(λ) is TD(0)
TEST_CASE("true_online_sarsa_lambda: lambda = 0 is sarsa and TD(0)", "[eligibility_traces][true_online]")
{
    // the same corridor episodes, played at random, some of them cut short after 5 moves
    // (agreeing to the last few bits only: α (δ + Q - Q_old) - α (Q - Q_old) is α δ, rounded differently)
    corridor game(3);
    std::mt19937 prng(4);
    auto agent = [&](const int &,
                     const std::vector<unsigned int> & legal) {
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        return legal[pick(prng)];
    };
    using S = corridor_types::state;
    using A = corridor_types::action;
    relearn::sarsa<S,A> one_step(0.3, 0.9);
    relearn::true_online_sarsa_lambda<S,A> zero(0.3, 0.9, 0.0);
    corridor_types::policy one_memory, zero_memory;
    for (int i = 0; i < 200; i++) {
        auto r = games::play(game, agent, i % 2 == 0 ? 5 : 1000);
        one_step(r.episode, one_memory, r.terminated);
        zero(r.episode, zero_memory, r.terminated);
    }
    for (int s = 0; s < 4; s++) {
        for (unsigned int a = 0; a < 2; a++) {
            REQUIRE(zero_memory.value(S(s), A(a)) == Approx(one_memory.value(S(s), A(a))).epsilon(1e-12));
        }
    }
    // true online TD(λ = 0) on the random walk is n-step TD with n = 1
    walk_type walk(19, -1, 1, 6);
    walk_types::policy td_memory, zero_td_memory;
    relearn::n_step_td<walk_types::state, walk_types::action> td(0.4, 1.0, 1);
    relearn::true_online_td_lambda<walk_types::state, walk_types::action> zero_td(0.4, 1.0, 0.0);
    for (int i = 0; i < 50; i++) {
        auto r = games::play(walk, only_action, 100000);
        td(r.episode, td_memory, r.terminated);
        zero_td(r.episode, zero_td_memory, r.terminated);
    }
    for (unsigned int s = 1; s <= 19; s++) {
        REQUIRE(zero_td_memory.value(walk_types::state(s), walk_types::action(0)) ==
                Approx(td_memory.value(walk_types::state(s), walk_types::action(0))).epsilon(1e-12).margin(1e-15));
    }
}

/// verifies when an episode goes on from one call to the next, and when a new one starts
TEST_CASE("true_online_sarsa_lambda: episodes cut short, and one call after another", "[eligibility_traces][true_online]")
{
    auto x = step {state(1, "x"), action("a")};
    auto y = step {state(0, "y"), action("b")};
    auto z = step {state(2, "z"), action("c")};
    // an episode started with start_episode() and learnt a step at a time, as `games::play_online` gives it,
    // is the episode learnt at once
    relearn::true_online_sarsa_lambda<state,action> stepwise(0.5, 0.9, 0.8);
    relearn::true_online_sarsa_lambda<state,action> at_once(0.5, 0.9, 0.8);
    policy stepwise_memory, at_once_memory;
    stepwise.start_episode();
    stepwise(episode {x, y}, stepwise_memory, false);
    stepwise(episode {y, z}, stepwise_memory, false);
    stepwise(episode {z}, stepwise_memory, true);
    at_once(episode {x, y, z}, at_once_memory, true);
    for (const auto & link : {x, y, z}) {
        REQUIRE(stepwise_memory.value(link.state, link.action) == at_once_memory.value(link.state, link.action));
    }
    REQUIRE(stepwise.traces.entries().size() == 3);
    // after the end of an episode, the next call is a new one, its traces starting at 0,
    // even if it starts with the very link the last one ended with
    stepwise(episode {z}, stepwise_memory, true);
    REQUIRE(stepwise.traces.entries().size() == 1);
    REQUIRE(stepwise.traces.trace(state("x"), action("a")) == Approx(0));
    // without start_episode(), the steps of an episode given one call at a time are learnt as episodes of their own
    relearn::true_online_sarsa_lambda<state,action> unopened(0.5, 0.9, 0.8);
    policy unopened_memory;
    unopened(episode {x, y}, unopened_memory, false);
    unopened(episode {y, z}, unopened_memory, false);
    REQUIRE(unopened.traces.entries().size() == 1);
    REQUIRE(unopened.traces.trace(state("x"), action("a")) == Approx(0));
    // an episode cut short does not learn its last link: (s1, b) only gives Q' to the step before it.
    // γ = λ = α = 1: δ = 1 + Q(s1, b) - Q(s0, a) = 4, so Q(s0, a) = 4, and (s1, b) has no trace
    relearn::true_online_sarsa_lambda<state,action> cut(1.0, 1.0, 1.0);
    policy memory;
    memory.update(state("s1"), action("b"), 3);
    cut(episode {{state(1, "s0"), action("a")}, {state(100, "s1"), action("b")}}, memory, false);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(4));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(3));
    REQUIRE(cut.traces.entries().size() == 1);
    REQUIRE(cut.q_old == Approx(3));
    // the next whole episode is a new one, even if it starts with the very link the one cut short ended with:
    // (s1, b) earns 5 and ends it, δ = 5 - 3 = 2, so Q(s1, b) = 5, and Q(s0, a) keeps its 4
    cut(episode {{state(5, "s1"), action("b")}}, memory, true);
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(5));
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(4));
    REQUIRE(cut.traces.entries().size() == 1);
    REQUIRE(cut.traces.trace(state("s0"), action("a")) == Approx(0));
    // and start_episode() starts a new one whatever the last call was
    cut.start_episode();
    cut(episode {{state(1, "s0"), action("a")}, {state(100, "s1"), action("b")}}, memory, false);
    cut.start_episode();
    REQUIRE(cut.traces.entries().empty());
    REQUIRE(cut.q_old == Approx(0));
}

/// verifies the random walk: true online TD(λ) learns at least as well as TD(λ), and much better with a large α
TEST_CASE("true_online_td_lambda: the 19-state walk, at least as good as TD(lambda), and steady with a large alpha", "[eligibility_traces][true_online][random_walk]")
{
    // the error over the first 10 walks, 50 runs, for α = 0.05 ... 1. Over 10 seeds (the same with libstdc++
    // and with libc++), at its best α true online TD(λ) was better than TD(λ) every time, by 0.007 to 0.010
    // (λ = 0.8) and 0.012 to 0.016 (λ = 0.9); with α = 1, its error was 0.353 to 0.371 (λ = 0.8) and 0.406 to 0.428
    // (λ = 0.9), against 0.656 to 0.705 and 0.962 to 0.968 for TD(λ), whose values grow out of bounds
    using td_type   = relearn::td_lambda<walk_types::state, walk_types::action>;
    using true_type = relearn::true_online_td_lambda<walk_types::state, walk_types::action>;
    for (double lambda : {0.8, 0.9}) {
        auto td = figure_12_6<td_type>(lambda, 1);
        auto true_online = figure_12_6<true_type>(lambda, 1);
        const double best_td = *std::min_element(td.begin(), td.end());
        const double best_true = *std::min_element(true_online.begin(), true_online.end());
        INFO("λ = " << lambda << ": best TD(λ) " << best_td << ", best true online " << best_true
             << "; with α = 1, TD(λ) " << td.back() << ", true online " << true_online.back());
        REQUIRE(best_true < best_td);
        REQUIRE(true_online.back() < 0.45);
        REQUIRE(true_online.back() < td.back() - 0.2);
    }
}

/// verifies true online Sarsa(λ) control, learning after every step: it learns the exact values of the ε-greedy policy
TEST_CASE("true_online_sarsa_lambda: online, learns q of the epsilon-greedy policy in the corridor", "[eligibility_traces][true_online][convergence]")
{
    auto exact = corridor::model(0.9);
    std::vector<std::vector<double>> pi(5, {0.05, 0.95});
    auto v_epsilon = exact.policy_values(pi);
    // over 10 seeds, with libstdc++ and with libc++, the worst errors were 0.023 to 0.123, for λ = 0.5 and 0.9
    for (double lambda : {0.5, 0.9}) {
        relearn::true_online_sarsa_lambda<corridor_types::state, corridor_types::action> learner(0.1, 0.9, lambda);
        auto memory = learn_corridor(learner, 1, false, true);
        INFO("λ = " << lambda);
        REQUIRE(worst_error(memory, exact, v_epsilon) < 0.2);
    }
}

/// verifies true online Sarsa(λ)'s constructor, and brace initialisation
TEST_CASE("true_online_sarsa_lambda: constructors and brace initialisation", "[eligibility_traces][true_online]")
{
    relearn::true_online_sarsa_lambda<state,action> defaults;
    REQUIRE(defaults.alpha == Approx(0.9));
    REQUIRE(defaults.gamma == Approx(0.9));
    REQUIRE(defaults.lambda == Approx(0.9));
    REQUIRE(defaults.q_old == Approx(0));
    REQUIRE(defaults.traces.entries().empty());
    auto learner = relearn::true_online_sarsa_lambda<state,action>{0.1, 1.0, 0.8};
    REQUIRE(learner.alpha == Approx(0.1));
    REQUIRE(learner.gamma == Approx(1));
    REQUIRE(learner.lambda == Approx(0.8));
    relearn::true_online_td_lambda<state,action> td(0.3, 0.5, 0.4);
    REQUIRE(td.lambda == Approx(0.4));
}

/********************************************************************************
 ***********   Truncated TD(λ), the online λ-return algorithm, variable λ and γ
 ********************************************************************************/

namespace
{
/**
 * @brief a random episode of 4 states and 2 actions, which comes back to the same pairs
 * @param prng draws the states, actions, rewards (-2 ... 3) and the length (1 ... 12)
 */
episode random_episode(std::mt19937 & prng)
{
    std::uniform_int_distribution<int> pick_state(0, 3);
    std::uniform_int_distribution<int> pick_action(0, 1);
    std::uniform_int_distribution<int> pick_reward(-2, 3);
    std::uniform_int_distribution<std::size_t> pick_length(1, 12);
    episode e;
    const std::size_t length = pick_length(prng);
    for (std::size_t i = 0; i < length; i++) {
        e.push_back({state(pick_reward(prng), "s" + std::to_string(pick_state(prng))),
                     action("a" + std::to_string(pick_action(prng)))});
    }
    return e;
}

/// @return random values for the 8 pairs of `random_episode`, -0.5 ... 0.5, drawn from @param prng
policy random_values(std::mt19937 & prng)
{
    std::uniform_real_distribution<double> uniform(0, 1);
    policy values;
    for (int s = 0; s < 4; s++) {
        for (int a = 0; a < 2; a++) {
            values.update(state("s" + std::to_string(s)), action("a" + std::to_string(a)), uniform(prng) - 0.5);
        }
    }
    return values;
}

/// @return the largest |@param a - @param b| over the 8 pairs of `random_episode`, or infinity if one is not a number
double largest_difference(const policy & a,
                          const policy & b)
{
    double largest = 0;
    for (int s = 0; s < 4; s++) {
        for (int k = 0; k < 2; k++) {
            state s_k("s" + std::to_string(s));
            action a_k("a" + std::to_string(k));
            const double d = std::fabs(a.value(s_k, a_k) - b.value(s_k, a_k));
            largest = std::isfinite(d) ? std::max(largest, d) : std::numeric_limits<double>::infinity();
        }
    }
    return largest;
}

/**
 * @brief truncated TD(λ) (Section 12.3), written from its definition, for one episode
 * @param e is the episode, @param start the values it starts from
 * @param alpha, @param gamma, @param lambda and @param n are α, γ, λ and the horizon
 * @param terminated is false for an episode cut short, whose last link is not a step
 * @param lambda_of and @param gamma_of, if they are set, give λ(s, a) and γ(s) instead (Section 12.8)
 * @return the values at every time h = 0, 1, ..., T, and last, once the episode has ended
 *
 * At time h: V_h, the value of (s_h, a_h), is read; then step t = h - n is learnt from
 * G^λ_{t:h} = (1 - λ) Σ_{k=1}^{h-t-1} λ^{k-1} G_{t:t+k} + λ^{h-t-1} G_{t:h},
 * G_{t:t+k} = r_t + ... + γ^{k-1} r_{t+k-1} + γ^k V_{t+k} (no V after the end of the episode).
 * With a λ and a γ that vary, from Section 12.8's G^λs_t = r_t + γ_{t+1} ((1 - λ_{t+1}) V_{t+1} + λ_{t+1} G^λs_{t+1}),
 * truncated at h: G^λs_{h-1:h} = r_{h-1} + γ_h V_h.
 * Once the episode has ended, the steps left are learnt in order, from G^λ_{t:T}.
 */
std::vector<policy> truncated_td_lambda(const episode & e,
                                        const policy & start,
                                        double alpha,
                                        double gamma,
                                        double lambda,
                                        std::size_t n,
                                        bool terminated,
                                        const std::function<double(const state &, const action &)> & lambda_of = nullptr,
                                        const std::function<double(const state &)> & gamma_of = nullptr)
{
    const std::size_t T = terminated ? e.size() : e.size() - 1;
    policy w = start;
    std::vector<double> V(e.size() + 1, 0);
    auto lambda_return = [&](std::size_t t, std::size_t h) {
        auto n_step = [&](std::size_t k) {
            double G = 0;
            for (std::size_t i = 0; i < k; i++) {
                G += std::pow(gamma, i) * e[t + i].state.reward();
            }
            if (t + k < e.size()) {
                G += std::pow(gamma, k) * V[t + k];
            }
            return G;
        };
        // a λ and a γ that vary: Section 12.8's recursion, truncated at h
        if (lambda_of) {
            std::function<double(std::size_t)> G = [&](std::size_t k) {
                double g = e[k].state.reward();
                if (k + 1 < e.size()) {
                    const double l = lambda_of(e[k + 1].state, e[k + 1].action);
                    const double y = gamma_of(e[k + 1].state);
                    g += k + 1 == h ? y * V[k + 1] : y * ((1 - l) * V[k + 1] + l * G(k + 1));
                }
                return g;
            };
            return G(t);
        }
        double G = 0;
        for (std::size_t k = 1; k + t < h; k++) {
            G += (1 - lambda) * std::pow(lambda, k - 1) * n_step(k);
        }
        return G + std::pow(lambda, h - t - 1) * n_step(h - t);
    };
    auto learn = [&](std::size_t t, std::size_t h) {
        double q = w.value(e[t].state, e[t].action);
        w.update(e[t].state, e[t].action, q + alpha * (lambda_return(t, h) - q));
    };
    std::vector<policy> history {w};
    for (std::size_t h = 1; h <= T; h++) {
        if (h < e.size()) {
            V[h] = w.value(e[h].state, e[h].action);
        }
        if (h >= n) {
            learn(h - n, h);
        }
        history.push_back(w);
    }
    if (terminated) {
        for (std::size_t t = T >= n ? T - n + 1 : 0; t < T; t++) {
            learn(t, T);
        }
        history.push_back(w);
    }
    return history;
}

/**
 * @brief the offline λ-return algorithm of the second edition (equation 12.4), written from its definition
 * @param e is a finished episode, @param memory holds the values
 * @param alpha, @param gamma and @param lambda are α, γ and λ
 *
 * G^λ_t = (1 - λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_{t:t+n} + λ^{T-t-1} G_t, every G_{t:t+n} with the values from before
 * the episode; then at its end, for t = 0 ... T-1 in order, Q(s_t,a_t) moves by α (G^λ_t - Q(s_t,a_t)),
 * each update seeing the ones before it.
 */
void sequential_lambda_return(const episode & e,
                              policy & memory,
                              double alpha,
                              double gamma,
                              double lambda)
{
    const auto before = memory;
    const std::size_t T = e.size();
    for (std::size_t t = 0; t < T; t++) {
        double G = 0;
        double rewards = 0;
        double discount = 1;
        double weight = 1;
        for (std::size_t k = t; k < T; k++) {
            rewards += discount * e[k].state.reward();
            discount *= gamma;
            if (k + 1 < T) {
                G += (1 - lambda) * weight * (rewards + discount * before.value(e[k + 1].state, e[k + 1].action));
                weight *= lambda;
            }
            else {
                G += weight * rewards;
            }
        }
        double q = memory.value(e[t].state, e[t].action);
        memory.update(e[t].state, e[t].action, q + alpha * (G - q));
    }
}

/**
 * @brief the λ-return algorithm offline with a λ and a γ that vary (Section 12.8), written from its definition
 * @param e is a finished episode, @param memory holds the values
 * @param alpha is α, @param lambda_of gives λ(s, a) and @param gamma_of γ(s)
 *
 * G^λs_t = r_t + γ_{t+1} ((1 - λ_{t+1}) Q(s_{t+1}, a_{t+1}) + λ_{t+1} G^λs_{t+1}), and G^λs_{T-1} = r_{T-1},
 * with γ_{t+1} = γ(s_{t+1}) and λ_{t+1} = λ(s_{t+1}, a_{t+1}); every Q(s_t, a_t) moves by α (G^λs_t - Q(s_t, a_t)),
 * all of them worked out with the values from before the episode.
 */
void variable_lambda_return(const episode & e,
                            policy & memory,
                            double alpha,
                            const std::function<double(const state &, const action &)> & lambda_of,
                            const std::function<double(const state &)> & gamma_of)
{
    const auto before = memory;
    const std::size_t T = e.size();
    std::vector<double> G(T, 0);
    for (std::size_t t = T; t-- > 0;) {
        G[t] = e[t].state.reward();
        if (t + 1 < T) {
            const auto & next = e[t + 1];
            const double l = lambda_of(next.state, next.action);
            G[t] += gamma_of(next.state) * ((1 - l) * before.value(next.state, next.action) + l * G[t + 1]);
        }
    }
    for (std::size_t t = 0; t < T; t++) {
        memory.update(e[t].state, e[t].action,
                      memory.value(e[t].state, e[t].action) + alpha * (G[t] - before.value(e[t].state, e[t].action)));
    }
}

/**
 * @brief learn @param e a step at a time, as `games::play_online` gives it: (s_t, a_t) and the next link,
 *        then the last step alone if the episode ended
 * @param learner is the learner, @param memory holds the values
 * @param terminated is false for an episode cut short, whose last link is not a step
 * @param after gets the values after every step
 */
template <class learner_type>
void step_by_step(learner_type & learner,
                  const episode & e,
                  policy & memory,
                  bool terminated,
                  std::vector<policy> & after)
{
    learner.start_episode();
    const std::size_t T = terminated ? e.size() : e.size() - 1;
    for (std::size_t t = 0; t < T; t++) {
        if (t + 1 < e.size()) {
            learner(episode {e[t], e[t + 1]}, memory, false);
        }
        else {
            learner(episode {e[t]}, memory, true);
        }
        after.push_back(memory);
    }
}

/**
 * @brief the lowest RMS error, over α = 0.05, 0.1, ... 1, of a learner on the 19-state walk over its first 10 walks
 * @param make makes the learner, given α
 * @param seed seeds the walks of the runs (seed * 1000 + run)
 * @return the lowest error, after each of the first 10 walks, averaged over them and over 50 runs
 */
template <class maker>
double lowest_walk_error(maker make,
                         unsigned int seed)
{
    double lowest = std::numeric_limits<double>::infinity();
    for (int i = 1; i <= 20; i++) {
        double error = 0;
        for (unsigned int run = 0; run < 50; run++) {
            walk_type walk(19, -1, 1, seed * 1000 + run);
            walk_types::policy memory;
            auto learner = make(0.05 * i);
            for (unsigned int k = 0; k < 10; k++) {
                auto r = games::play(walk, only_action, 100000);
                learner(r.episode, memory, r.terminated);
                error += rms_error(walk, memory) / 500;
            }
        }
        lowest = std::min(lowest, error);
    }
    return lowest;
}
} // namespace

/// verifies the truncated λ-returns of a record, worked out by hand
TEST_CASE("lambda_return_record: truncated lambda-returns worked out by hand", "[eligibility_traces][truncated]")
{
    // r = 1, 2, 3; γ_1 = 0.5, λ_1 = 0.5, V_1 = 2; γ_2 = 1, λ_2 = 0.25, V_2 = 4; γ_3 = 0.9, λ_3 = 0, V_3 = 10
    relearn::lambda_return_record<state,action> record;
    record.push_back({state("x"), action("a"), 1, 0.5, 0.5, 2});
    record.push_back({state("y"), action("a"), 2, 1, 0.25, 4});
    record.push_back({state("z"), action("a"), 3, 0.9, 0, 10});
    REQUIRE(record.size() == 3);
    // G_{2:3} = 3 + 0.9 * 10 = 12, G_{1:3} = 2 + (0.75 * 4 + 0.25 * 12) = 8, G_{0:3} = 1 + 0.5 (0.5 * 2 + 0.5 * 8) = 3.5
    auto G = record.returns(0, 3);
    REQUIRE(G.size() == 3);
    REQUIRE(G[0] == Approx(3.5));
    REQUIRE(G[1] == Approx(8));
    REQUIRE(G[2] == Approx(12));
    // truncated at 2: G_{1:2} = 2 + 4 = 6, G_{0:2} = 1 + 0.5 (0.5 * 2 + 0.5 * 6) = 3
    G = record.returns(0, 2);
    REQUIRE(G.size() == 2);
    REQUIRE(G[0] == Approx(3));
    REQUIRE(G[1] == Approx(6));
    G = record.returns(1, 3);
    REQUIRE(G.size() == 2);
    REQUIRE(G[0] == Approx(8));
    REQUIRE(record.returns(2, 2).empty());
    record.clear();
    REQUIRE(record.size() == 0);
}

/// verifies truncated Sarsa(λ) on one episode, worked out by hand
TEST_CASE("truncated_sarsa_lambda: values worked out by hand", "[eligibility_traces][truncated]")
{
    // (x, 1), (y, 0), (x, 2), from Q(x) = 1, Q(y) = 2, with α = γ = λ = 0.5 and n = 2
    // h = 1: V_1 = Q(y) = 2.  h = 2: V_2 = Q(x) = 1, read before step 0 is learnt: G_{1:2} = 0 + 0.5 * 1 = 0.5,
    // G_{0:2} = 1 + 0.5 (0.5 * 2 + 0.5 * 0.5) = 1.625, Q(x) = 1 + 0.5 (1.625 - 1) = 1.3125.
    // h = 3, the end: G_{2:3} = 2, G_{1:3} = 0.5 (0.5 * 1 + 0.5 * 2) = 0.75, Q(y) = 2 + 0.5 (0.75 - 2) = 1.375;
    // then step 2: Q(x) = 1.3125 + 0.5 (2 - 1.3125) = 1.65625
    const episode e {{state(1, "x"), action("a")}, {state(0, "y"), action("a")}, {state(2, "x"), action("a")}};
    policy start;
    start.update(state("x"), action("a"), 1);
    start.update(state("y"), action("a"), 2);
    relearn::truncated_sarsa_lambda<state,action> learner(0.5, 0.5, 0.5, 2);
    policy memory = start;
    learner(e, memory);
    REQUIRE(memory.value(state("x"), action("a")) == Approx(1.65625));
    REQUIRE(memory.value(state("y"), action("a")) == Approx(1.375));
    // a step at a time: nothing is learnt after the first step, step 0 after the second, and the rest at the end
    std::vector<policy> after;
    memory = start;
    step_by_step(learner, e, memory, true, after);
    REQUIRE(after[0].value(state("x"), action("a")) == Approx(1));
    REQUIRE(after[1].value(state("x"), action("a")) == Approx(1.3125));
    REQUIRE(after[1].value(state("y"), action("a")) == Approx(2));
    REQUIRE(after[2].value(state("x"), action("a")) == Approx(1.65625));
    REQUIRE(after[2].value(state("y"), action("a")) == Approx(1.375));
    // cut short after the second step: (x, 2) is only the next link, so step 0 alone is learnt
    memory = start;
    learner(episode {e[0], e[1], e[2]}, memory, false);
    REQUIRE(memory.value(state("x"), action("a")) == Approx(1.3125));
    REQUIRE(memory.value(state("y"), action("a")) == Approx(2));
}

/// verifies the online λ-return algorithm on one episode, worked out by hand
TEST_CASE("online_lambda_return: values worked out by hand", "[eligibility_traces][online_lambda_return]")
{
    // (x, 1), (y, 0), (x, 2), from Q(x) = 1, Q(y) = 2, with α = γ = λ = 0.5
    // h = 1: V_1 = 2, G_{0:1} = 1 + 0.5 * 2 = 2, Q(x) = 1.5.
    // h = 2: V_2 = Q(x) in w_1 = 1.5, G_{1:2} = 0.75, G_{0:2} = 1 + 0.5 (0.5 * 2 + 0.5 * 0.75) = 1.6875; from w_0,
    //        Q(x) = 1 + 0.5 * 0.6875 = 1.34375, then Q(y) = 2 + 0.5 (0.75 - 2) = 1.375.
    // h = 3: G_{2:3} = 2, G_{1:3} = 0.5 (0.5 * 1.5 + 0.5 * 2) = 0.875, G_{0:3} = 1 + 0.5 (1 + 0.4375) = 1.71875;
    //        from w_0, Q(x) = 1.359375, Q(y) = 1.4375, then Q(x) = 1.359375 + 0.5 (2 - 1.359375) = 1.6796875
    const episode e {{state(1, "x"), action("a")}, {state(0, "y"), action("a")}, {state(2, "x"), action("a")}};
    policy start;
    start.update(state("x"), action("a"), 1);
    start.update(state("y"), action("a"), 2);
    relearn::online_lambda_return<state,action> learner(0.5, 0.5, 0.5);
    std::vector<policy> after;
    policy memory = start;
    step_by_step(learner, e, memory, true, after);
    REQUIRE(after[0].value(state("x"), action("a")) == Approx(1.5));
    REQUIRE(after[0].value(state("y"), action("a")) == Approx(2));
    REQUIRE(after[1].value(state("x"), action("a")) == Approx(1.34375));
    REQUIRE(after[1].value(state("y"), action("a")) == Approx(1.375));
    REQUIRE(after[2].value(state("x"), action("a")) == Approx(1.6796875));
    REQUIRE(after[2].value(state("y"), action("a")) == Approx(1.4375));
    // in one call, the same
    memory = start;
    learner(e, memory);
    REQUIRE(memory.value(state("x"), action("a")) == Approx(1.6796875));
    REQUIRE(memory.value(state("y"), action("a")) == Approx(1.4375));
    // a pair that is not in the episode does not move
    REQUIRE(memory.value(state("z"), action("a")) == 0);
}

/// verifies Sarsa(λ) with a λ and a γ that vary from state to state, worked out by hand
TEST_CASE("sarsa_lambda: a lambda and a gamma for every state, worked out by hand", "[eligibility_traces][variable]")
{
    // (x, 1), (y, 0), (x, 2), from Q(x) = 1, Q(y) = 2, with α = 0.5, λ(x) = 0.5, λ(y) = 1, γ(x) = 0.5, γ(y) = 1
    // t = 0: δ = 1 + γ(y) Q(y) - Q(x) = 2, z(x) = 1, Q(x) = 2.
    // t = 1: δ = 0 + γ(x) Q(x) - Q(y) = -1, the traces fade by γ(y) λ(y) = 1: z(x) = 1, z(y) = 1; Q(x) = 1.5, Q(y) = 1.5.
    // t = 2: δ = 2 - Q(x) = 0.5, the traces fade by γ(x) λ(x) = 0.25: z(x) = 1.25, z(y) = 0.25;
    //        Q(x) = 1.5 + 0.25 * 1.25 = 1.8125, Q(y) = 1.5 + 0.25 * 0.25 = 1.5625
    const episode e {{state(1, "x"), action("a")}, {state(0, "y"), action("a")}, {state(2, "x"), action("a")}};
    relearn::sarsa_lambda<state,action> learner(0.5, 0.9, 0.9);
    learner.lambda_of = [](const state & s, const action &) { return s.trait() == "x" ? 0.5 : 1.0; };
    learner.gamma_of = [](const state & s) { return s.trait() == "x" ? 0.5 : 1.0; };
    policy memory;
    memory.update(state("x"), action("a"), 1);
    memory.update(state("y"), action("a"), 2);
    learner(e, memory);
    REQUIRE(memory.value(state("x"), action("a")) == Approx(1.8125));
    REQUIRE(memory.value(state("y"), action("a")) == Approx(1.5625));
    REQUIRE(learner.traces.trace(state("x"), action("a")) == Approx(1.25));
    REQUIRE(learner.traces.trace(state("y"), action("a")) == Approx(0.25));
    // the same λ and γ for every state is the constant one, to the last bit
    relearn::sarsa_lambda<state,action> constant(0.5, 0.7, 0.6);
    relearn::sarsa_lambda<state,action> as_functions(0.5, 0.1, 0.1);
    as_functions.lambda_of = [](const state &, const action &) { return 0.6; };
    as_functions.gamma_of = [](const state &) { return 0.7; };
    policy a, b;
    constant(e, a);
    as_functions(e, b);
    REQUIRE(a.value(state("x"), action("a")) == b.value(state("x"), action("a")));
    REQUIRE(a.value(state("y"), action("a")) == b.value(state("y"), action("a")));
}

/// verifies truncated Sarsa(λ) against its definition, after every step, with random episodes, α, γ, λ and n, and a λ and a γ that vary
TEST_CASE("truncated_sarsa_lambda: after every step, it is truncated TD(lambda) written from its definition", "[eligibility_traces][truncated][variable]")
{
    std::mt19937 prng(6);
    std::uniform_real_distribution<double> uniform(0, 1);
    std::uniform_int_distribution<unsigned int> pick_n(1, 6);
    double largest = 0;
    for (int trial = 0; trial < 300; trial++) {
        const double alpha  = 0.05 + 0.9 * uniform(prng);
        const double gamma  = trial % 5 == 0 ? 1.0 : uniform(prng);
        const double lambda = trial % 7 == 0 ? 1.0 : (trial % 11 == 0 ? 0.0 : uniform(prng));
        const unsigned int n = pick_n(prng);
        const bool terminated = trial % 3 != 0;
        const episode e = random_episode(prng);
        const policy start = random_values(prng);
        // every other trial with a random λ(s, a) and γ(s), γ = 0 in a fifth of the states
        std::map<std::string, double> lambdas, gammas;
        for (int s = 0; s < 4; s++) {
            gammas["s" + std::to_string(s)] = uniform(prng) < 0.2 ? 0.0 : uniform(prng);
            for (int a = 0; a < 2; a++) {
                lambdas["s" + std::to_string(s) + "a" + std::to_string(a)] = uniform(prng);
            }
        }
        std::function<double(const state &, const action &)> lambda_of;
        std::function<double(const state &)> gamma_of;
        if (trial % 2 == 1) {
            lambda_of = [lambdas](const state & s, const action & a) { return lambdas.at(s.trait() + a.trait()); };
            gamma_of = [gammas](const state & s) { return gammas.at(s.trait()); };
        }
        const auto expected = truncated_td_lambda(e, start, alpha, gamma, lambda, n, terminated, lambda_of, gamma_of);
        // a step at a time: after step t, the values at time t + 1, and at the end those once it ended
        relearn::truncated_sarsa_lambda<state,action> learner(alpha, gamma, lambda, n);
        learner.lambda_of = lambda_of;
        learner.gamma_of = gamma_of;
        std::vector<policy> after;
        policy memory = start;
        step_by_step(learner, e, memory, terminated, after);
        for (std::size_t t = 0; t < after.size(); t++) {
            const bool last = terminated && t + 1 == after.size();
            largest = std::max(largest, largest_difference(after[t], last ? expected.back() : expected[t + 1]));
        }
        // and in one call, the values at the end
        relearn::truncated_sarsa_lambda<state,action> whole(alpha, gamma, lambda, n);
        whole.lambda_of = lambda_of;
        whole.gamma_of = gamma_of;
        policy at_once = start;
        whole(e, at_once, terminated);
        largest = std::max(largest, largest_difference(at_once, expected.back()));
    }
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 1e-12);
}

/// verifies truncated Sarsa(λ)'s equivalences: n = 1 is SARSA, λ = 1 is n-step SARSA, and a long n the offline λ-return
TEST_CASE("truncated_sarsa_lambda: n = 1 is sarsa, lambda = 1 is n-step sarsa, a long n the offline lambda-return", "[eligibility_traces][truncated]")
{
    std::mt19937 prng(7);
    std::uniform_real_distribution<double> uniform(0, 1);
    double largest = 0;
    for (int trial = 0; trial < 200; trial++) {
        const double alpha = 0.05 + 0.9 * uniform(prng);
        const double gamma = trial % 5 == 0 ? 1.0 : uniform(prng);
        const double lambda = uniform(prng);
        const bool terminated = trial % 3 != 0;
        const episode e = random_episode(prng);
        const policy start = random_values(prng);
        // n = 1, whatever λ is: SARSA
        relearn::truncated_sarsa_lambda<state,action> one(alpha, gamma, lambda, 1);
        relearn::sarsa<state,action> sarsa(alpha, gamma);
        policy a = start, b = start;
        one(e, a, terminated);
        sarsa(e, b, terminated);
        largest = std::max(largest, largest_difference(a, b));
        // λ = 1: n-step SARSA, with the same n
        const unsigned int n = 1 + trial % 5;
        relearn::truncated_sarsa_lambda<state,action> full(alpha, gamma, 1.0, n);
        relearn::n_step_sarsa<state,action> n_step(alpha, gamma, n);
        a = start;
        b = start;
        full(e, a, terminated);
        n_step(e, b, terminated);
        largest = std::max(largest, largest_difference(a, b));
        // n at least the length of the episode: the offline λ-return algorithm of equation 12.4
        if (terminated) {
            relearn::truncated_sarsa_lambda<state,action> late(alpha, gamma, lambda, 12);
            a = start;
            b = start;
            late(e, a);
            sequential_lambda_return(e, b, alpha, gamma, lambda);
            largest = std::max(largest, largest_difference(a, b));
        }
    }
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 1e-12);
}

/// verifies the online λ-return algorithm against its definition, and true online Sarsa(λ) against it, after every step
TEST_CASE("online_lambda_return: after every step, its definition, and true online Sarsa(lambda) with a lambda and a gamma that vary", "[eligibility_traces][online_lambda_return][variable]")
{
    // random episodes, α, γ and λ; every other trial with a random λ(s, a) and γ(s), γ = 0 in a fifth of the states
    std::mt19937 prng(8);
    std::uniform_real_distribution<double> uniform(0, 1);
    double from_definition = 0;
    double true_online = 0;
    for (int trial = 0; trial < 300; trial++) {
        const double alpha  = 0.05 + 0.9 * uniform(prng);
        const double gamma  = trial % 5 == 0 ? 1.0 : uniform(prng);
        const double lambda = trial % 7 == 0 ? 1.0 : uniform(prng);
        const bool terminated = trial % 3 != 0;
        const bool variable = trial % 2 == 1;
        const episode e = random_episode(prng);
        const policy start = random_values(prng);
        std::map<std::string, double> lambdas, gammas;
        for (int s = 0; s < 4; s++) {
            gammas["s" + std::to_string(s)] = uniform(prng) < 0.2 ? 0.0 : uniform(prng);
            for (int a = 0; a < 2; a++) {
                lambdas["s" + std::to_string(s) + "a" + std::to_string(a)] = uniform(prng);
            }
        }
        relearn::online_lambda_return<state,action> learner(alpha, gamma, lambda);
        relearn::true_online_sarsa_lambda<state,action> online(alpha, gamma, lambda);
        if (variable) {
            auto lambda_of = [lambdas](const state & s, const action & a) { return lambdas.at(s.trait() + a.trait()); };
            auto gamma_of = [gammas](const state & s) { return gammas.at(s.trait()); };
            learner.lambda_of = lambda_of;
            learner.gamma_of = gamma_of;
            online.lambda_of = lambda_of;
            online.gamma_of = gamma_of;
        }
        // a copy for the one call below: an episode cut short a step at a time stays open
        auto whole = learner;
        std::vector<policy> after, true_after;
        policy memory = start, true_memory = start;
        step_by_step(learner, e, memory, terminated, after);
        step_by_step(online, e, true_memory, terminated, true_after);
        // the constant λ and γ against the definition in `online_lambda_return` above, w_{t+1} after step t
        const auto expected = online_lambda_return(e, start, alpha, gamma, lambda, terminated);
        for (std::size_t t = 0; t < after.size(); t++) {
            if (!variable) {
                from_definition = std::max(from_definition, largest_difference(after[t], expected[t + 1]));
            }
            true_online = std::max(true_online, largest_difference(after[t], true_after[t]));
        }
        // in one call, the same values at the end
        policy at_once = start;
        whole(e, at_once, terminated);
        true_online = std::max(true_online, largest_difference(at_once, memory));
    }
    INFO("from the definition, the largest difference is " << from_definition);
    INFO("from true online Sarsa(λ), the largest difference is " << true_online);
    REQUIRE(from_definition < 1e-12);
    REQUIRE(true_online < 1e-12);
}

/// verifies the online λ-return algorithm's equivalences: λ = 0 is SARSA, and λ = 1 ends with constant-α Monte Carlo
TEST_CASE("online_lambda_return: lambda = 0 is sarsa, lambda = 1 ends as monte carlo", "[eligibility_traces][online_lambda_return]")
{
    std::mt19937 prng(9);
    std::uniform_real_distribution<double> uniform(0, 1);
    double largest = 0;
    for (int trial = 0; trial < 200; trial++) {
        const double alpha = 0.05 + 0.9 * uniform(prng);
        const double gamma = trial % 5 == 0 ? 1.0 : uniform(prng);
        const bool terminated = trial % 3 != 0;
        const episode e = random_episode(prng);
        const policy start = random_values(prng);
        // λ = 0: SARSA, after every step
        relearn::online_lambda_return<state,action> zero(alpha, gamma, 0.0);
        relearn::sarsa<state,action> sarsa(alpha, gamma);
        std::vector<policy> after;
        policy a = start;
        step_by_step(zero, e, a, terminated, after);
        policy b = start;
        const std::size_t T = terminated ? e.size() : e.size() - 1;
        for (std::size_t t = 0; t < T; t++) {
            if (t + 1 < e.size()) {
                sarsa(episode {e[t], e[t + 1]}, b, false);
            }
            else {
                sarsa(episode {e[t]}, b, true);
            }
            largest = std::max(largest, largest_difference(after[t], b));
        }
        // λ = 1, once the episode has ended: every step from its return, in order (n-step SARSA with n ≥ T)
        if (terminated) {
            relearn::online_lambda_return<state,action> one(alpha, gamma, 1.0);
            relearn::n_step_sarsa<state,action> monte_carlo(alpha, gamma, 12);
            a = start;
            b = start;
            one(e, a);
            monte_carlo(e, b);
            largest = std::max(largest, largest_difference(a, b));
        }
    }
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 1e-12);
}

/// verifies offline TD(λ) with a λ and a γ that vary: it is still exactly the offline λ-return, the one of Section 12.8
TEST_CASE("sarsa_lambda: offline with a lambda and a gamma that vary, it is the lambda-return of Section 12.8", "[eligibility_traces][variable]")
{
    std::mt19937 prng(10);
    std::uniform_real_distribution<double> uniform(0, 1);
    double largest = 0;
    for (int trial = 0; trial < 300; trial++) {
        const double alpha = 0.05 + 0.9 * uniform(prng);
        const episode e = random_episode(prng);
        const policy start = random_values(prng);
        std::map<std::string, double> lambdas, gammas;
        for (int s = 0; s < 4; s++) {
            gammas["s" + std::to_string(s)] = uniform(prng) < 0.2 ? 0.0 : (uniform(prng) < 0.2 ? 1.0 : uniform(prng));
            for (int a = 0; a < 2; a++) {
                lambdas["s" + std::to_string(s) + "a" + std::to_string(a)] = uniform(prng) < 0.2 ? 1.0 : uniform(prng);
            }
        }
        std::function<double(const state &, const action &)> lambda_of =
            [lambdas](const state & s, const action & a) { return lambdas.at(s.trait() + a.trait()); };
        std::function<double(const state &)> gamma_of = [gammas](const state & s) { return gammas.at(s.trait()); };
        relearn::td_lambda<state,action> td(alpha, 0.9, 0.9);
        td.offline = true;
        td.lambda_of = lambda_of;
        td.gamma_of = gamma_of;
        policy a = start, b = start;
        td(e, a);
        variable_lambda_return(e, b, alpha, lambda_of, gamma_of);
        largest = std::max(largest, largest_difference(a, b));
    }
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 1e-12);
}

/// verifies that γ = 0 in a state ends the return there: an episode learnt whole is the two halves learnt one after the other
TEST_CASE("sarsa_lambda and the online lambda-return: gamma = 0 in a state splits the episode there", "[eligibility_traces][variable]")
{
    std::mt19937 prng(11);
    std::uniform_real_distribution<double> uniform(0, 1);
    double largest = 0;
    int split = 0;
    for (int trial = 0; trial < 300; trial++) {
        const double alpha = 0.05 + 0.9 * uniform(prng);
        const double gamma = uniform(prng);
        const double lambda = uniform(prng);
        const episode e = random_episode(prng);
        const policy start = random_values(prng);
        // γ = 0 in s0, γ elsewhere: the episode is cut before its first visit of s0 after the start, if any
        auto gamma_of = [gamma](const state & s) { return s.trait() == "s0" ? 0.0 : gamma; };
        std::size_t c = 1;
        while (c < e.size() && !(e[c].state == state("s0"))) {
            c++;
        }
        if (c == e.size()) {
            continue;
        }
        split++;
        const episode first(e.begin(), e.begin() + static_cast<std::ptrdiff_t>(c));
        const episode second(e.begin() + static_cast<std::ptrdiff_t>(c), e.end());
        // Sarsa(λ), accumulating, and the online λ-return algorithm
        relearn::sarsa_lambda<state,action> whole(alpha, gamma, lambda);
        relearn::sarsa_lambda<state,action> halves(alpha, gamma, lambda);
        whole.gamma_of = gamma_of;
        halves.gamma_of = gamma_of;
        policy a = start, b = start;
        whole(e, a);
        halves(first, b);
        halves(second, b);
        largest = std::max(largest, largest_difference(a, b));
        relearn::online_lambda_return<state,action> online_whole(alpha, gamma, lambda);
        relearn::online_lambda_return<state,action> online_halves(alpha, gamma, lambda);
        online_whole.gamma_of = gamma_of;
        online_halves.gamma_of = gamma_of;
        a = start;
        b = start;
        online_whole(e, a);
        online_halves(first, b);
        online_halves(second, b);
        largest = std::max(largest, largest_difference(a, b));
    }
    INFO("the largest difference is " << largest << ", over " << split << " episodes");
    REQUIRE(split > 100);
    REQUIRE(largest < 1e-12);
}

/// verifies the cut-short and call-after-call rules of truncated Sarsa(λ) and the online λ-return algorithm
TEST_CASE("truncated_sarsa_lambda and online_lambda_return: episodes cut short, and one call after another", "[eligibility_traces][truncated][online_lambda_return]")
{
    const episode e {{state(1, "x"), action("a")}, {state(0, "y"), action("a")}, {state(2, "x"), action("a")}};
    // the largest difference of Q(x, a) and Q(y, a)
    auto differ = [](const policy & a, const policy & b) {
        return std::max(std::fabs(a.value(state("x"), action("a")) - b.value(state("x"), action("a"))),
                        std::fabs(a.value(state("y"), action("a")) - b.value(state("y"), action("a"))));
    };
    // an episode left open (cut short a step at a time) goes on until start_episode(); any other call is an episode
    relearn::truncated_sarsa_lambda<state,action> truncated(0.5, 0.5, 0.5, 2);
    relearn::online_lambda_return<state,action> online(0.5, 0.5, 0.5);
    policy a, b;
    truncated.start_episode();
    online.start_episode();
    truncated(episode {e[0], e[1]}, a, false);
    online(episode {e[0], e[1]}, b, false);
    // a new episode, started properly: the steps of the open one are forgotten
    truncated.start_episode();
    online.start_episode();
    policy a2, b2;
    truncated(e, a2);
    online(e, b2);
    policy a3, b3;
    relearn::truncated_sarsa_lambda<state,action> fresh_truncated(0.5, 0.5, 0.5, 2);
    relearn::online_lambda_return<state,action> fresh_online(0.5, 0.5, 0.5);
    fresh_truncated(e, a3);
    fresh_online(e, b3);
    REQUIRE(differ(a2, a3) == 0);
    REQUIRE(differ(b2, b3) == 0);
    // after a whole episode, the next call is a new one: the same episode twice is learnt twice the same way
    policy once_a, once_b;
    fresh_truncated(e, once_a);
    fresh_online(e, once_b);
    relearn::truncated_sarsa_lambda<state,action> twice_truncated(0.5, 0.5, 0.5, 2);
    relearn::online_lambda_return<state,action> twice_online(0.5, 0.5, 0.5);
    policy twice_a, twice_b;
    twice_truncated(e, twice_a);
    twice_online(e, twice_b);
    twice_truncated(e, twice_a);
    twice_online(e, twice_b);
    REQUIRE(differ(once_a, twice_a) > 0);
    // two calls of one episode each: the second starts afresh, from the values the first left
    policy check_a, check_b;
    relearn::truncated_sarsa_lambda<state,action> one_more(0.5, 0.5, 0.5, 2);
    relearn::online_lambda_return<state,action> one_more_online(0.5, 0.5, 0.5);
    one_more(e, check_a);
    one_more_online(e, check_b);
    one_more.start_episode();
    one_more_online.start_episode();
    one_more(e, check_a);
    one_more_online(e, check_b);
    REQUIRE(differ(check_a, twice_a) == 0);
    REQUIRE(differ(check_b, twice_b) == 0);
    // an episode learnt a step at a time to its end: the next call is a new episode again
    policy stepped_a, stepped_b;
    std::vector<policy> after;
    relearn::truncated_sarsa_lambda<state,action> stepped(0.5, 0.5, 0.5, 2);
    relearn::online_lambda_return<state,action> stepped_online(0.5, 0.5, 0.5);
    step_by_step(stepped, e, stepped_a, true, after);
    step_by_step(stepped_online, e, stepped_b, true, after);
    stepped(e, stepped_a);
    stepped_online(e, stepped_b);
    REQUIRE(differ(stepped_a, twice_a) == 0);
    REQUIRE(differ(stepped_b, twice_b) == 0);
    // and after it, a whole episode cut short then another whole one are learnt as by a fresh learner: the ended
    // episode does not stay open (the cut-short one would otherwise be carried on into the next call)
    relearn::true_online_sarsa_lambda<state,action> stepped_true(0.5, 0.5, 0.5);
    policy stepped_c;
    step_by_step(stepped_true, e, stepped_c, true, after);
    policy fresh_a = stepped_a, fresh_b = stepped_b, fresh_c = stepped_c;
    stepped(e, stepped_a, false);
    stepped(e, stepped_a);
    stepped_online(e, stepped_b, false);
    stepped_online(e, stepped_b);
    stepped_true(e, stepped_c, false);
    stepped_true(e, stepped_c);
    relearn::truncated_sarsa_lambda<state,action> new_truncated(0.5, 0.5, 0.5, 2);
    relearn::online_lambda_return<state,action> new_online(0.5, 0.5, 0.5);
    relearn::true_online_sarsa_lambda<state,action> new_true(0.5, 0.5, 0.5);
    new_truncated(e, fresh_a, false);
    new_truncated(e, fresh_a);
    new_online(e, fresh_b, false);
    new_online(e, fresh_b);
    new_true(e, fresh_c, false);
    new_true(e, fresh_c);
    REQUIRE(differ(stepped_a, fresh_a) == 0);
    REQUIRE(differ(stepped_b, fresh_b) == 0);
    REQUIRE(differ(stepped_c, fresh_c) == 0);
    // an empty episode learns nothing
    policy empty;
    truncated(episode {}, empty);
    online(episode {}, empty);
    REQUIRE(empty.value(state("x"), action("a")) == 0);
}

/// verifies the 19-state walk: a short horizon is best for truncated TD(λ), and online beats offline
TEST_CASE("truncated_td_lambda and online_lambda_return: the 19-state walk", "[eligibility_traces][truncated][online_lambda_return][random_walk]")
{
    using S = walk_types::state;
    using A = walk_types::action;
    // the lowest error over α, over the first 10 walks (50 runs). Over 10 seeds, with libstdc++ and with libc++, with
    // λ = 0.9: n = 1 (TD(0)) 0.341 to 0.348, n = 4 0.249 to 0.265, n = 1000 (the offline λ-return of equation 12.4)
    // 0.270 to 0.292, and in every one n = 4 was lower than n = 1 by 0.082 or more, than n = 1000 by 0.018 or more
    auto truncated = [](double lambda, unsigned int n) {
        return [lambda, n](double alpha) { return relearn::truncated_td_lambda<S,A>(alpha, 1.0, lambda, n); };
    };
    const double one   = lowest_walk_error(truncated(0.9, 1), 1);
    const double four  = lowest_walk_error(truncated(0.9, 4), 1);
    const double whole = lowest_walk_error(truncated(0.9, 1000), 1);
    INFO("λ = 0.9: n = 1 " << one << ", n = 4 " << four << ", n = 1000 " << whole);
    REQUIRE(four < one - 0.05);
    REQUIRE(four < whole - 0.01);
    // with λ = 0.8, the online λ-return algorithm was 0.242 to 0.263, and lower than the offline one by 0.006 or more
    const double offline = lowest_walk_error(truncated(0.8, 1000), 1);
    const double online  = lowest_walk_error([](double alpha) { return relearn::online_lambda_return<S,A>(alpha, 1.0, 0.8); }, 1);
    INFO("λ = 0.8: offline " << offline << ", online " << online);
    REQUIRE(online < offline - 0.003);
}

/// verifies a γ that varies: with γ = 0 in state 5, the walk is learnt up to its next visit of state 5
TEST_CASE("td_lambda, true_online_td_lambda and truncated_td_lambda: gamma = 0 in a state of the 19-state walk", "[eligibility_traces][variable][random_walk]")
{
    using S = walk_types::state;
    using A = walk_types::action;
    // the return stops when the walk reaches state 5: right of it, the value is the chance of leaving on the right
    // first, (s - 5) / 15; left of it, minus the chance of leaving on the left first, -(5 - s) / 5; and in 5, -1/15
    auto truth = [](unsigned int s) { return s > 5 ? (s - 5.0) / 15 : (s < 5 ? -(5.0 - s) / 5 : -1.0 / 15); };
    auto gamma_of = [](const S & s) { return s.trait() == 5 ? 0.0 : 1.0; };
    // 1000 walks with α = 0.02 and λ = 0.8 (n = 4). Over 10 seeds, with libstdc++ and with libc++, the mean RMS
    // error of 10 runs was 0.031 to 0.038 (TD(λ)), 0.030 to 0.038 (true online) and 0.026 to 0.032 (truncated)
    double errors[3] = {0, 0, 0};
    for (unsigned int run = 0; run < 10; run++) {
        for (int which = 0; which < 3; which++) {
            walk_type walk(19, -1, 1, 1000 + run);
            walk_types::policy memory;
            relearn::td_lambda<S,A> td(0.02, 1.0, 0.8);
            relearn::true_online_td_lambda<S,A> true_online(0.02, 1.0, 0.8);
            relearn::truncated_td_lambda<S,A> truncated(0.02, 1.0, 0.8, 4);
            td.gamma_of = gamma_of;
            true_online.gamma_of = gamma_of;
            truncated.gamma_of = gamma_of;
            for (int k = 0; k < 1000; k++) {
                auto r = games::play(walk, only_action, 100000);
                if (which == 0) {
                    td(r.episode, memory, r.terminated);
                }
                else if (which == 1) {
                    true_online(r.episode, memory, r.terminated);
                }
                else {
                    truncated(r.episode, memory, r.terminated);
                }
            }
            double sum = 0;
            for (unsigned int s = 1; s <= 19; s++) {
                const double d = memory.value(S(s), A(0)) - truth(s);
                sum += d * d;
            }
            errors[which] += std::sqrt(sum / 19) / 10;
        }
    }
    INFO("TD(λ) " << errors[0] << ", true online " << errors[1] << ", truncated " << errors[2]);
    REQUIRE(errors[0] < 0.05);
    REQUIRE(errors[1] < 0.05);
    REQUIRE(errors[2] < 0.05);
}

/// verifies truncated Sarsa(λ) and the online λ-return algorithm in control, learning after every step
TEST_CASE("truncated_sarsa_lambda and online_lambda_return: online, learn q of the epsilon-greedy policy in the corridor", "[eligibility_traces][truncated][online_lambda_return][convergence]")
{
    auto exact = corridor::model(0.9);
    std::vector<std::vector<double>> pi(5, {0.05, 0.95});
    auto v_epsilon = exact.policy_values(pi);
    using S = corridor_types::state;
    using A = corridor_types::action;
    // λ = 0.9 (n = 4). Over 10 seeds, with libstdc++ and with libc++, the worst errors were 0.036 to 0.103
    // (truncated) and 0.031 to 0.123 (the online λ-return algorithm)
    relearn::truncated_sarsa_lambda<S,A> truncated(0.1, 0.9, 0.9, 4);
    auto learnt = learn_corridor(truncated, 1, false, true);
    INFO("truncated: " << worst_error(learnt, exact, v_epsilon));
    REQUIRE(worst_error(learnt, exact, v_epsilon) < 0.2);
    relearn::online_lambda_return<S,A> online(0.1, 0.9, 0.9);
    learnt = learn_corridor(online, 1, false, true);
    INFO("online λ-return: " << worst_error(learnt, exact, v_epsilon));
    REQUIRE(worst_error(learnt, exact, v_epsilon) < 0.2);
}

/// verifies the constructors of truncated Sarsa(λ) and the online λ-return algorithm, and brace initialisation
TEST_CASE("truncated_sarsa_lambda and online_lambda_return: constructors and brace initialisation", "[eligibility_traces][truncated][online_lambda_return]")
{
    relearn::truncated_sarsa_lambda<state,action> defaults;
    REQUIRE(defaults.alpha == Approx(0.9));
    REQUIRE(defaults.gamma == Approx(0.9));
    REQUIRE(defaults.lambda == Approx(0.9));
    REQUIRE(defaults.n == 4);
    REQUIRE(!defaults.lambda_of);
    REQUIRE(!defaults.gamma_of);
    auto truncated = relearn::truncated_sarsa_lambda<state,action>{0.1, 1.0, 0.8, 6};
    REQUIRE(truncated.alpha == Approx(0.1));
    REQUIRE(truncated.gamma == Approx(1));
    REQUIRE(truncated.lambda == Approx(0.8));
    REQUIRE(truncated.n == 6);
    relearn::truncated_td_lambda<state,action> td(0.3, 0.5, 0.4, 2);
    REQUIRE(td.n == 2);
    relearn::online_lambda_return<state,action> online_defaults;
    REQUIRE(online_defaults.alpha == Approx(0.9));
    REQUIRE(online_defaults.gamma == Approx(0.9));
    REQUIRE(online_defaults.lambda == Approx(0.9));
    auto online = relearn::online_lambda_return<state,action>{0.2, 0.7, 0.3};
    REQUIRE(online.alpha == Approx(0.2));
    REQUIRE(online.gamma == Approx(0.7));
    REQUIRE(online.lambda == Approx(0.3));
}
