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
 * Tests of the eligibility traces (Sutton & Barto, Chapter 12): `sarsa_lambda`, `td_lambda`
 * and `watkins_q_lambda`:
 * - traces and values worked out by hand, accumulating and replacing, and Watkins's cut
 * - λ = 0 is `sarsa`, `q_learning` and TD(0), and offline TD(λ) is the offline λ-return algorithm
 * - the 19-state random walk (Figure 12.6): an intermediate λ is best
 * - control: the exact values in a slippery corridor, and the Dyna maze after one episode, and early on
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
 * @brief Figure 12.6: the lowest RMS error of TD(λ) on the 19-state walk, over its first 10 episodes
 * @param lambda is λ
 * @param seed seeds the walks of the runs (seed * 1000 + run)
 * @return the lowest, over α = 0.05, 0.1, ... 1, of the error after each of the first 10 walks,
 *         averaged over them and over 50 runs
 */
double best_of_figure_12_6(double lambda,
                           unsigned int seed)
{
    double best = std::numeric_limits<double>::infinity();
    for (int i = 1; i <= 20; i++) {
        double error = 0;
        for (unsigned int run = 0; run < 50; run++) {
            walk_type walk(19, -1, 1, seed * 1000 + run);
            walk_types::policy memory;
            relearn::td_lambda<walk_types::state, walk_types::action> td(0.05 * i, 1.0, lambda);
            for (unsigned int k = 0; k < 10; k++) {
                auto r = games::play(walk, only_action, 100000);
                td(r.episode, memory, r.terminated);
                error += rms_error(walk, memory) / 500;
            }
        }
        best = std::min(best, error);
    }
    return best;
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
 * @brief learn the corridor, one whole episode at a time
 * @param learner is the learner to train
 * @param seed seeds the corridor and the agent
 * @param random_play is true for an agent that moves at random, false for ε-greedy on what it learns (ε = 0.1)
 * @return the learnt policy, after 20000 episodes, with α = 0.1 * 1000 / (1000 + k) in episode k
 */
template <class learner_type>
corridor_types::policy learn_corridor(learner_type & learner,
                                      unsigned int seed,
                                      bool random_play)
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
        auto r = games::play(game, agent, 1000);
        learner(r.episode, memory, r.terminated);
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
