// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/game.hpp"
#include "../examples/games/frozen_lake.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <string>
#include "solvers.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of the off-policy eligibility traces (Sutton & Barto, Sections 12.9 and 12.10): `off_policy_td_lambda`,
 * `off_policy_expected_sarsa_lambda` and `tree_backup_lambda`:
 * - values and traces worked out by hand, and a greedy π cutting the traces of Tree-Backup(λ)
 * - offline, each one makes exactly the changes of its λ-return, written from the book's definitions, with a λ and
 *   a γ that vary too; λ = 0 is Expected SARSA, π = b is TD(λ), λ = 1 is n-step Q(σ) and n-step tree backup, and
 *   Tree-Backup(λ) is off-policy Expected Sarsa(λ b) when b is uniform
 * - control and prediction from random play, against exact values: the slippery corridor and FrozenLake
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;
using distribution = std::vector<std::pair<action,double>>;

/**
 * @brief a random episode of 4 states and 2 actions, which comes back to the same pairs
 * @param prng draws the states, actions, rewards (-2 ... 3) and the length (1 ... 12)
 * @param distinct is true for an episode that never comes back to a state (at most 4 steps)
 */
episode random_episode(std::mt19937 & prng,
                       bool distinct = false)
{
    std::uniform_int_distribution<int> pick_state(0, 3);
    std::uniform_int_distribution<int> pick_action(0, 1);
    std::uniform_int_distribution<int> pick_reward(-2, 3);
    std::uniform_int_distribution<std::size_t> pick_length(1, distinct ? 4 : 12);
    std::vector<int> order = {0, 1, 2, 3};
    std::shuffle(order.begin(), order.end(), prng);
    episode e;
    const std::size_t length = pick_length(prng);
    for (std::size_t i = 0; i < length; i++) {
        const int s = distinct ? order[i] : pick_state(prng);
        e.push_back({state(pick_reward(prng), "s" + std::to_string(s)),
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

/// @return |@param a - @param b| / max(1, |@param b|), or infinity if it is not a number
double relative(double a,
                double b)
{
    const double d = std::fabs(a - b) / std::max(1.0, std::fabs(b));
    return std::isfinite(d) ? d : std::numeric_limits<double>::infinity();
}

/**
 * @return the largest difference of @param a and @param b over the 8 pairs of `random_episode`, relative to the
 *         value of @param b if it is above 1 (the ratios of importance sampling can make the values large)
 */
double largest_difference(const policy & a,
                          const policy & b)
{
    double largest = 0;
    for (int s = 0; s < 4; s++) {
        for (int k = 0; k < 2; k++) {
            state s_k("s" + std::to_string(s));
            action a_k("a" + std::to_string(k));
            largest = std::max(largest, relative(a.value(s_k, a_k), b.value(s_k, a_k)));
        }
    }
    return largest;
}

/**
 * @struct policies
 * @brief a random π and b over the 4 states and 2 actions of `random_episode`, and a random λ(s, a) and γ(s)
 */
struct policies
{
    std::map<std::string, double> pi;
    std::map<std::string, double> b;
    std::map<std::string, double> lambdas;
    std::map<std::string, double> gammas;

    /// @brief draw them all from @param prng, with γ = 0 in a fifth of the states, and π = 0 for a tenth of the pairs
    explicit policies(std::mt19937 & prng)
    {
        std::uniform_real_distribution<double> uniform(0, 1);
        for (int s = 0; s < 4; s++) {
            const std::string name = "s" + std::to_string(s);
            const double p = uniform(prng) < 0.2 ? (uniform(prng) < 0.5 ? 0.0 : 1.0) : uniform(prng);
            pi[name + "a0"] = p;
            pi[name + "a1"] = 1 - p;
            const double q = 0.1 + 0.8 * uniform(prng);
            b[name + "a0"] = q;
            b[name + "a1"] = 1 - q;
            gammas[name] = uniform(prng) < 0.2 ? 0.0 : uniform(prng);
            for (int a = 0; a < 2; a++) {
                lambdas[name + "a" + std::to_string(a)] = uniform(prng);
            }
        }
    }
    /// @return π(a | s)
    std::function<double(const state &, const action &)> target() const
    {
        auto copy = pi;
        return [copy](const state & s, const action & a) { return copy.at(s.trait() + a.trait()); };
    }
    /// @return π(· | s), as a distribution
    std::function<distribution(const state &)> probabilities() const
    {
        auto copy = pi;
        return [copy](const state & s) {
            return distribution {{action("a0"), copy.at(s.trait() + "a0")}, {action("a1"), copy.at(s.trait() + "a1")}};
        };
    }
    /// @return b(a | s)
    std::function<double(const state &, const action &)> behaviour() const
    {
        auto copy = b;
        return [copy](const state & s, const action & a) { return copy.at(s.trait() + a.trait()); };
    }
    /// @return λ(s, a)
    std::function<double(const state &, const action &)> lambda_of() const
    {
        auto copy = lambdas;
        return [copy](const state & s, const action & a) { return copy.at(s.trait() + a.trait()); };
    }
    /// @return γ(s)
    std::function<double(const state &)> gamma_of() const
    {
        auto copy = gammas;
        return [copy](const state & s) { return copy.at(s.trait()); };
    }
};

/**
 * @brief the state-value λ-return of Section 12.9, offline, written from its definition, for one episode
 * @param e is the episode, @param V the state values (V(s) is kept as the value of (s, "v")), updated at its end
 * @param alpha is α, @param pi gives π(a | s), @param b gives b(a | s), @param lambda_of λ(s, a) and @param gamma_of γ(s)
 * @param terminated is false for an episode cut short, whose last link is not a step
 *
 * G^λs_t = ρ_t (r_t + γ_{t+1} ((1 - λ_{t+1}) V(s_{t+1}) + λ_{t+1} G^λs_{t+1})) + (1 - ρ_t) V(s_t), ρ_t = π(a_t | s_t) / b(a_t | s_t),
 * with nothing after the end of an episode, and G^λs_T = V(s_T) for the state an episode cut short reached;
 * every V(s_t) moves by α (G^λs_t - V(s_t)), all of them worked out with the values from before the episode.
 */
void state_lambda_return(const episode & e,
                         policy & V,
                         double alpha,
                         const std::function<double(const state &, const action &)> & pi,
                         const std::function<double(const state &, const action &)> & b,
                         const std::function<double(const state &, const action &)> & lambda_of,
                         const std::function<double(const state &)> & gamma_of,
                         bool terminated)
{
    const auto before = V;
    const action v("v");
    const std::size_t T = terminated ? e.size() : e.size() - 1;
    std::vector<double> G(T + 1, 0);
    if (!terminated) {
        G[T] = before.value(e[T].state, v);
    }
    for (std::size_t t = T; t-- > 0;) {
        const double rho = pi(e[t].state, e[t].action) / b(e[t].state, e[t].action);
        double after = e[t].state.reward();
        if (t + 1 < e.size()) {
            const auto & next = e[t + 1];
            const double l = lambda_of(next.state, next.action);
            after += gamma_of(next.state) * ((1 - l) * before.value(next.state, v) + l * G[t + 1]);
        }
        G[t] = rho * after + (1 - rho) * before.value(e[t].state, v);
    }
    for (std::size_t t = 0; t < T; t++) {
        V.update(e[t].state, v, V.value(e[t].state, v) + alpha * (G[t] - before.value(e[t].state, v)));
    }
}

/**
 * @brief the action-value λ-returns of Sections 12.9 and 12.10, offline, written from their definitions, for one episode
 * @param e is the episode, @param Q the values, updated at its end
 * @param alpha is α, @param pi gives π(a | s), @param b gives b(a | s) (not used by tree backup)
 * @param lambda_of gives λ(s, a) and @param gamma_of γ(s)
 * @param terminated is false for an episode cut short, whose last link is not a step
 * @param tree_backup is true for Tree-Backup(λ), false for importance sampling with control variates
 *
 * With V̄(s) = Σ_a π(a | s) Q(s, a), importance sampling (Section 12.9):
 * G^λa_t = r_t + γ_{t+1} ((1 - λ_{t+1}) V̄(s_{t+1}) + λ_{t+1} (ρ_{t+1} G^λa_{t+1} + V̄(s_{t+1}) - ρ_{t+1} Q(s_{t+1}, a_{t+1}))),
 * and tree backup (Section 12.10):
 * G^λa_t = r_t + γ_{t+1} ((1 - λ_{t+1}) V̄(s_{t+1}) + λ_{t+1} (Σ_{a ≠ a_{t+1}} π(a | s_{t+1}) Q(s_{t+1}, a) + π(a_{t+1} | s_{t+1}) G^λa_{t+1})),
 * with nothing after the end of an episode, and G^λa_T = Q(s_T, a_T) for the link an episode cut short ends with;
 * every Q(s_t, a_t) moves by α (G^λa_t - Q(s_t, a_t)), all of them worked out with the values from before the episode.
 */
void action_lambda_return(const episode & e,
                          policy & Q,
                          double alpha,
                          const std::function<double(const state &, const action &)> & pi,
                          const std::function<double(const state &, const action &)> & b,
                          const std::function<double(const state &, const action &)> & lambda_of,
                          const std::function<double(const state &)> & gamma_of,
                          bool terminated,
                          bool tree_backup)
{
    const auto before = Q;
    const std::size_t T = terminated ? e.size() : e.size() - 1;
    auto expected = [&](const state & s) {
        return pi(s, action("a0")) * before.value(s, action("a0")) + pi(s, action("a1")) * before.value(s, action("a1"));
    };
    std::vector<double> G(T + 1, 0);
    if (!terminated) {
        G[T] = before.value(e[T].state, e[T].action);
    }
    for (std::size_t t = T; t-- > 0;) {
        G[t] = e[t].state.reward();
        if (t + 1 < e.size()) {
            const auto & next = e[t + 1];
            const double l = lambda_of(next.state, next.action);
            double rest = 0;
            if (tree_backup) {
                const action other(next.action.trait() == "a0" ? "a1" : "a0");
                rest = pi(next.state, other) * before.value(next.state, other) + pi(next.state, next.action) * G[t + 1];
            }
            else {
                const double rho = pi(next.state, next.action) / b(next.state, next.action);
                rest = rho * G[t + 1] + expected(next.state) - rho * before.value(next.state, next.action);
            }
            G[t] += gamma_of(next.state) * ((1 - l) * expected(next.state) + l * rest);
        }
    }
    for (std::size_t t = 0; t < T; t++) {
        Q.update(e[t].state, e[t].action, Q.value(e[t].state, e[t].action) + alpha * (G[t] - before.value(e[t].state, e[t].action)));
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
 * @brief learn the corridor from random play, one whole episode at a time
 * @param learner is the learner to train
 * @param seed seeds the corridor and the agent
 * @return the learnt policy, after 20000 episodes, with α = 0.1 * 1000 / (1000 + k) in episode k
 */
template <class learner_type>
corridor_types::policy learn_corridor(learner_type & learner,
                                      unsigned int seed)
{
    corridor game(seed);
    std::mt19937 prng(seed + 100);
    corridor_types::policy memory;
    auto agent = [&](const int &,
                     const std::vector<unsigned int> & legal) {
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        return legal[pick(prng)];
    };
    for (int k = 1; k <= 20000; k++) {
        learner.alpha = 0.1 * 1000.0 / (1000.0 + k);
        auto r = games::play(game, agent, 1000);
        learner(r.episode, memory, r.terminated);
    }
    return memory;
}

using lake_type  = games::frozen_lake;
using lake_types = games::types<lake_type>;

/**
 * @brief the RMS error of state values learnt on slippery 4x4 FrozenLake from random play, over its 11 cells that are
 *        not a hole or the goal
 * @param lambda is λ, @param seed seeds the runs (seed * 1000 + run)
 * @return the mean error, over 5 runs of 10000 episodes, of off-policy TD(λ) learning the values of a policy that takes
 *         the optimal move with 0.85 and every other with 0.05 (γ = 0.95, α = 0.02)
 */
double lake_prediction_error(double lambda,
                             unsigned int seed)
{
    lake_type probe(lake_type::map_4x4(), true);
    auto model = probe.model();
    relearn::dynamic_programming<lake_types::state, lake_types::action> optimal;
    optimal.gamma = 0.95;
    optimal.theta = 1e-12;
    optimal.value_iteration(model);
    // the optimal move of each cell, then π and its exact values
    auto best = [&](unsigned int s) {
        unsigned int a = 0;
        for (unsigned int k = 1; k < 4; k++) {
            if (optimal.q_value(model, lake_types::state(s), lake_types::action(k)) >
                optimal.q_value(model, lake_types::state(s), lake_types::action(a)) + 1e-12) {
                a = k;
            }
        }
        return a;
    };
    auto pi = [&](const lake_types::state & s, const lake_types::action & a) { return a.trait() == best(s.trait()) ? 0.85 : 0.05; };
    auto b = [](const lake_types::state &, const lake_types::action &) { return 0.25; };
    relearn::dynamic_programming<lake_types::state, lake_types::action> exact;
    exact.gamma = 0.95;
    exact.theta = 1e-12;
    exact.evaluate(model, pi);
    double error = 0;
    for (unsigned int run = 0; run < 5; run++) {
        lake_type lake(lake_type::map_4x4(), true, seed * 1000 + run);
        std::mt19937 prng(seed * 1000 + run + 500);
        std::uniform_int_distribution<unsigned int> pick(0, 3);
        auto agent = [&](const unsigned int &, const std::vector<unsigned int> &) { return pick(prng); };
        relearn::off_policy_td_lambda<lake_types::state, lake_types::action> td(0.02, 0.95, lambda, pi, b);
        for (unsigned int k = 0; k < 10000; k++) {
            auto r = games::play(lake, agent, lake.step_limit());
            td(r.episode, r.terminated);
        }
        double sum = 0;
        double cells = 0;
        for (unsigned int s = 0; s < 16; s++) {
            if (probe.tile(s) != 'H' && probe.tile(s) != 'G') {
                const double d = td.value(lake_types::state(s)) - exact.value(lake_types::state(s));
                sum += d * d;
                cells++;
            }
        }
        const double rms = std::sqrt(sum / cells);
        error += (std::isfinite(rms) ? std::min(rms, 1000.0) : 1000.0) / 5;
    }
    return error;
}
} // namespace

/// verifies off-policy TD(λ) on one episode, worked out by hand
TEST_CASE("off_policy_td_lambda: values and traces worked out by hand", "[off_policy_traces][off_policy_td]")
{
    // π(a | x) = 0.8, π(b | x) = 0.2, π(a | y) = 0.4, π(b | y) = 0.6, b = 1/2, α = γ = λ = 0.5;
    // the episode (x, a, 1), (y, b, 2), (x, b, 4), so ρ = 1.6, 1.2 and 0.4.
    // t = 0: δ = 1, z(x) = 1.6, V(x) = 0.8.
    // t = 1: δ = 2 + 0.5 * 0.8 - 0 = 2.4, z(x) = 1.2 * 0.25 * 1.6 = 0.48, z(y) = 1.2; V(x) = 1.376, V(y) = 1.44.
    // t = 2: δ = 4 - 1.376 = 2.624, z(x) = 0.4 * (0.25 * 0.48 + 1) = 0.448, z(y) = 0.4 * 0.25 * 1.2 = 0.12;
    //        V(x) = 1.376 + 0.5 * 2.624 * 0.448 = 1.963776, V(y) = 1.44 + 0.5 * 2.624 * 0.12 = 1.59744
    const std::map<std::string, double> pi = {{"xa", 0.8}, {"xb", 0.2}, {"ya", 0.4}, {"yb", 0.6}};
    auto target = [pi](const state & s, const action & a) { return pi.at(s.trait() + a.trait()); };
    auto b = [](const state &, const action &) { return 0.5; };
    relearn::off_policy_td_lambda<state,action> td(0.5, 0.5, 0.5, target, b);
    td(episode {{state(1, "x"), action("a")}, {state(2, "y"), action("b")}, {state(4, "x"), action("b")}});
    REQUIRE(td.value(state("x")) == Approx(1.963776));
    REQUIRE(td.value(state("y")) == Approx(1.59744));
    REQUIRE(td.traces.trace(state("x"), relearn::action<int>(0)) == Approx(0.448));
    REQUIRE(td.traces.trace(state("y"), relearn::action<int>(0)) == Approx(0.12));
    REQUIRE(td.value(state("never")) == 0);
    // a step π would never take (ρ = 0) cuts every trace, and leaves the value of its state where it was
    const std::map<std::string, double> never_b = {{"xa", 1.0}, {"xb", 0.0}, {"ya", 0.4}, {"yb", 0.6}};
    relearn::off_policy_td_lambda<state,action> cut(0.5, 0.5, 0.5, [never_b](const state & s, const action & a) { return never_b.at(s.trait() + a.trait()); }, b);
    cut(episode {{state(1, "x"), action("a")}, {state(2, "y"), action("b")}, {state(4, "x"), action("b")}});
    REQUIRE(cut.traces.entries().empty());
    // t = 0: ρ = 2, z(x) = 2, V(x) = 1; t = 1: δ = 2 + 0.5 - 0 = 2.5, z(x) = 1.2 * 0.5 = 0.6, z(y) = 1.2,
    // V(x) = 1 + 0.75 = 1.75, V(y) = 1.5; t = 2: ρ = 0, nothing moves
    REQUIRE(cut.value(state("x")) == Approx(1.75));
    REQUIRE(cut.value(state("y")) == Approx(1.5));
}

/// verifies off-policy Expected Sarsa(λ) and Tree-Backup(λ) on one episode, worked out by hand
TEST_CASE("off_policy_expected_sarsa_lambda and tree_backup_lambda: values and traces worked out by hand", "[off_policy_traces][tree_backup][expected_sarsa]")
{
    // the same episode, π and b. V̄(x) = 0.8 Q(x, a) + 0.2 Q(x, b), V̄(y) = 0.4 Q(y, a) + 0.6 Q(y, b).
    // t = 0: δ = 1, z(x, a) = 1, Q(x, a) = 0.5.  t = 1: δ = 2 + 0.5 * V̄(x) = 2.2.  t = 2: δ = 4 - Q(x, b) = 4.
    // Importance sampling, the traces fade by γλρ = 0.3, then 0.1:
    //   z(x, a) = 0.3, Q(x, a) = 0.5 + 0.5 * 2.2 * 0.3 = 0.83, Q(y, b) = 1.1;
    //   z(x, a) = 0.03, z(y, b) = 0.1, Q(x, a) = 0.89, Q(y, b) = 1.3, Q(x, b) = 2
    // Tree backup, they fade by γλπ = 0.15, then 0.05:
    //   z(x, a) = 0.15, Q(x, a) = 0.665, Q(y, b) = 1.1;  z(x, a) = 0.0075, z(y, b) = 0.05, Q(x, a) = 0.68, Q(y, b) = 1.2, Q(x, b) = 2
    const episode e {{state(1, "x"), action("a")}, {state(2, "y"), action("b")}, {state(4, "x"), action("b")}};
    const std::map<std::string, double> pi = {{"xa", 0.8}, {"xb", 0.2}, {"ya", 0.4}, {"yb", 0.6}};
    auto probabilities = [pi](const state & s) {
        return distribution {{action("a"), pi.at(s.trait() + "a")}, {action("b"), pi.at(s.trait() + "b")}};
    };
    auto b = [](const state &, const action &) { return 0.5; };
    relearn::off_policy_expected_sarsa_lambda<state,action> sampled(0.5, 0.5, 0.5, probabilities, b);
    policy q;
    sampled(e, q);
    REQUIRE(q.value(state("x"), action("a")) == Approx(0.89));
    REQUIRE(q.value(state("y"), action("b")) == Approx(1.3));
    REQUIRE(q.value(state("x"), action("b")) == Approx(2));
    REQUIRE(sampled.traces.trace(state("x"), action("a")) == Approx(0.03));
    REQUIRE(sampled.traces.trace(state("y"), action("b")) == Approx(0.1));
    relearn::tree_backup_lambda<state,action> tree(0.5, 0.5, 0.5, probabilities);
    policy p;
    tree(e, p);
    REQUIRE(p.value(state("x"), action("a")) == Approx(0.68));
    REQUIRE(p.value(state("y"), action("b")) == Approx(1.2));
    REQUIRE(p.value(state("x"), action("b")) == Approx(2));
    REQUIRE(tree.traces.trace(state("x"), action("a")) == Approx(0.0075));
    REQUIRE(tree.traces.trace(state("y"), action("b")) == Approx(0.05));
    // the traces of importance sampling grow when π takes an action more often than b: λ = 1, (x, a) twice
    relearn::off_policy_expected_sarsa_lambda<state,action> grows(0.5, 1.0, 1.0, probabilities, b);
    policy g;
    grows(episode {{state(0, "x"), action("a")}, {state(0, "x"), action("a")}}, g);
    REQUIRE(grows.traces.trace(state("x"), action("a")) == Approx(1.6 + 1));
}

/// verifies that a greedy π cuts the traces of Tree-Backup(λ) after a move that is not greedy
TEST_CASE("tree_backup_lambda: a greedy pi cuts the traces after a move that is not greedy", "[off_policy_traces][tree_backup]")
{
    // greedy π (not set), Q(y, a) = 1 to start with, α = γ = λ = 0.5, the same episode:
    // t = 0: V̄(y) = 1, δ = 1 + 0.5 = 1.5, Q(x, a) = 0.75.
    // t = 1: V̄(x) = 0.75, δ = 2 + 0.375 = 2.375; b is not greedy in y, so π(b | y) = 0 and the trace of (x, a) is cut:
    //        Q(y, b) = 1.1875, and Q(x, a) stays 0.75.
    // t = 2: δ = 4, b is not greedy in x either: Q(x, b) = 2, and Q(y, b) stays 1.1875
    relearn::tree_backup_lambda<state,action> tree(0.5, 0.5, 0.5);
    policy q;
    q.update(state("y"), action("a"), 1);
    tree(episode {{state(1, "x"), action("a")}, {state(2, "y"), action("b")}, {state(4, "x"), action("b")}}, q);
    REQUIRE(q.value(state("x"), action("a")) == Approx(0.75));
    REQUIRE(q.value(state("y"), action("b")) == Approx(1.1875));
    REQUIRE(q.value(state("x"), action("b")) == Approx(2));
    REQUIRE(q.value(state("y"), action("a")) == Approx(1));
    REQUIRE(tree.traces.entries().size() == 1);
    REQUIRE(tree.traces.trace(state("x"), action("b")) == Approx(1));
}

/// verifies that a greedy π offline is read from the values from before the episode, as everything else is
TEST_CASE("off_policy_expected_sarsa_lambda and tree_backup_lambda: offline, a greedy pi is the one from before the episode", "[off_policy_traces][tree_backup][expected_sarsa]")
{
    // greedy π, b = 1/2, Q(y, a) = 1 to start with, α = γ = λ = 0.5, the episode (y, b, 3), (y, b, 2), (y, b, 1).
    // Offline, every δ and π use the values from before the episode, where a is greedy in y (b is not stored yet):
    // t = 0: δ = 3 + 0.5 * 1 = 3.5, Q(y, b) = 1.75.  t = 1: δ = 2 + 0.5 * 1 = 2.5, π(b | y) = 0 cuts the trace,
    // Q(y, b) = 1.75 + 1.25 = 3.  t = 2: δ = 1, cut again, Q(y, b) = 3.5.
    // Online, b is greedy in y after the first step, so the trace of (y, b) is kept, and Q(y, b) ends elsewhere
    // (1.19921875 with importance sampling, ρ = 2, and 1.49951171875 with tree backup, worked out in Python)
    const episode e {{state(3, "y"), action("b")}, {state(2, "y"), action("b")}, {state(1, "y"), action("b")}};
    auto b = [](const state &, const action &) { return 0.5; };
    for (bool offline : {true, false}) {
        relearn::off_policy_expected_sarsa_lambda<state,action> sampled(0.5, 0.5, 0.5, nullptr, b);
        relearn::tree_backup_lambda<state,action> tree(0.5, 0.5, 0.5);
        sampled.offline = tree.offline = offline;
        policy q, p;
        q.update(state("y"), action("a"), 1);
        p.update(state("y"), action("a"), 1);
        sampled(e, q);
        tree(e, p);
        REQUIRE(q.value(state("y"), action("b")) == Approx(offline ? 3.5 : 1.19921875));
        REQUIRE(p.value(state("y"), action("b")) == Approx(offline ? 3.5 : 1.49951171875));
        REQUIRE(q.value(state("y"), action("a")) == Approx(1));
    }
}

/// verifies that offline, each learner makes exactly the changes of its λ-return, written from the book's definitions
TEST_CASE("off-policy traces: offline, they are the lambda-returns of Sections 12.9 and 12.10", "[off_policy_traces]")
{
    // 300 trials of 5 random episodes each, learnt one after the other, with random α, γ, λ, π and b (π = 0 or 1 in a
    // fifth of the states), a third of the episodes cut short, and every other trial with a random λ(s, a) and γ(s)
    std::mt19937 prng(5);
    std::uniform_real_distribution<double> uniform(0, 1);
    double largest = 0;
    for (int trial = 0; trial < 300; trial++) {
        const double alpha  = 0.05 + 0.9 * uniform(prng);
        const double gamma  = trial % 5 == 0 ? 1.0 : uniform(prng);
        const double lambda = trial % 7 == 0 ? 1.0 : uniform(prng);
        const policies random(prng);
        std::function<double(const state &, const action &)> lambda_of = [lambda](const state &, const action &) { return lambda; };
        std::function<double(const state &)> gamma_of = [gamma](const state &) { return gamma; };
        relearn::off_policy_td_lambda<state,action> td(alpha, gamma, lambda, random.target(), random.behaviour());
        relearn::off_policy_expected_sarsa_lambda<state,action> sampled(alpha, gamma, lambda, random.probabilities(), random.behaviour());
        relearn::tree_backup_lambda<state,action> tree(alpha, gamma, lambda, random.probabilities());
        td.offline = sampled.offline = tree.offline = true;
        if (trial % 2 == 1) {
            lambda_of = random.lambda_of();
            gamma_of = random.gamma_of();
            td.lambda_of = sampled.lambda_of = tree.lambda_of = lambda_of;
            td.gamma_of = sampled.gamma_of = tree.gamma_of = gamma_of;
        }
        policy V, Q_sampled = random_values(prng), Q_tree = Q_sampled;
        policy expected_sampled = Q_sampled, expected_tree = Q_tree;
        for (int k = 0; k < 5; k++) {
            const episode e = random_episode(prng);
            const bool terminated = (trial + k) % 3 != 0;
            td(e, terminated);
            state_lambda_return(e, V, alpha, random.target(), random.behaviour(), lambda_of, gamma_of, terminated);
            sampled(e, Q_sampled, terminated);
            action_lambda_return(e, expected_sampled, alpha, random.target(), random.behaviour(), lambda_of, gamma_of, terminated, false);
            tree(e, Q_tree, terminated);
            action_lambda_return(e, expected_tree, alpha, random.target(), random.behaviour(), lambda_of, gamma_of, terminated, true);
            for (int s = 0; s < 4; s++) {
                const state s_k("s" + std::to_string(s));
                largest = std::max(largest, relative(td.value(s_k), V.value(s_k, action("v"))));
            }
            largest = std::max(largest, largest_difference(Q_sampled, expected_sampled));
            largest = std::max(largest, largest_difference(Q_tree, expected_tree));
        }
    }
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 1e-12);
}

/// verifies the equivalences: λ = 0 is Expected SARSA, π = b is TD(λ), and Tree-Backup(λ) is importance sampling with λ b
TEST_CASE("off-policy traces: lambda = 0 is expected sarsa, pi = b is TD(lambda), and a uniform b turns tree backup into importance sampling", "[off_policy_traces]")
{
    std::mt19937 prng(6);
    std::uniform_real_distribution<double> uniform(0, 1);
    double largest = 0;
    for (int trial = 0; trial < 200; trial++) {
        const double alpha = 0.05 + 0.9 * uniform(prng);
        const double gamma = trial % 5 == 0 ? 1.0 : uniform(prng);
        const double lambda = uniform(prng);
        const bool terminated = trial % 3 != 0;
        const policies random(prng);
        const episode e = random_episode(prng);
        const policy start = random_values(prng);
        // λ = 0: Expected SARSA, for both
        relearn::expected_sarsa<state,action> expected(alpha, gamma, random.probabilities());
        relearn::off_policy_expected_sarsa_lambda<state,action> sampled(alpha, gamma, 0.0, random.probabilities(), random.behaviour());
        relearn::tree_backup_lambda<state,action> tree(alpha, gamma, 0.0, random.probabilities());
        policy a = start, b = start, c = start;
        expected(e, a, terminated);
        sampled(e, b, terminated);
        tree(e, c, terminated);
        largest = std::max(largest, largest_difference(a, b));
        largest = std::max(largest, largest_difference(a, c));
        // π = b: every ρ is 1, and off-policy TD(λ) is TD(λ) on the states alone
        relearn::off_policy_td_lambda<state,action> td(alpha, gamma, lambda, random.behaviour(), random.behaviour());
        relearn::td_lambda<state,action> on_policy(alpha, gamma, lambda);
        episode states_only;
        for (const auto & link : e) {
            states_only.push_back({link.state, action("v")});
        }
        td(e, terminated);
        policy V;
        on_policy(states_only, V, terminated);
        for (int s = 0; s < 4; s++) {
            const state s_k("s" + std::to_string(s));
            largest = std::max(largest, relative(td.value(s_k), V.value(s_k, action("v"))));
        }
        // b = 1/2 for both actions: tree backup's γλπ is importance sampling's γ (λ/2) (π/b), online and offline
        auto uniform_b = [](const state &, const action &) { return 0.5; };
        for (bool offline : {false, true}) {
            relearn::tree_backup_lambda<state,action> by_tree(alpha, gamma, lambda, random.probabilities());
            relearn::off_policy_expected_sarsa_lambda<state,action> by_ratio(alpha, gamma, lambda / 2, random.probabilities(), uniform_b);
            by_tree.offline = by_ratio.offline = offline;
            a = start;
            b = start;
            by_tree(e, a, terminated);
            by_ratio(e, b, terminated);
            largest = std::max(largest, largest_difference(a, b));
        }
    }
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 1e-12);
}

/// verifies that with λ = 1, offline, they are n-step Q(σ) with σ = 1 and n-step tree backup to the end of the episode
TEST_CASE("off-policy traces: lambda = 1 offline is n-step Q(sigma) with sigma = 1, and n-step tree backup", "[off_policy_traces]")
{
    // on episodes that never come back to a state, the n-step learners' updates, made one after the other,
    // never change a value a later update reads, so they make the same changes as the offline λ-returns
    std::mt19937 prng(7);
    std::uniform_real_distribution<double> uniform(0, 1);
    double largest = 0;
    for (int trial = 0; trial < 200; trial++) {
        const double alpha = 0.05 + 0.9 * uniform(prng);
        const double gamma = trial % 5 == 0 ? 1.0 : uniform(prng);
        const policies random(prng);
        const episode e = random_episode(prng, true);
        const policy start = random_values(prng);
        relearn::off_policy_expected_sarsa_lambda<state,action> sampled(alpha, gamma, 1.0, random.probabilities(), random.behaviour());
        relearn::n_step_q_sigma<state,action> q_sigma(alpha, gamma, 12, 1.0, random.probabilities());
        q_sigma.behaviour = random.behaviour();
        relearn::tree_backup_lambda<state,action> tree(alpha, gamma, 1.0, random.probabilities());
        relearn::n_step_tree_backup<state,action> n_step_tree(alpha, gamma, 12, random.probabilities());
        sampled.offline = tree.offline = true;
        policy a = start, b = start, c = start, d = start;
        sampled(e, a);
        q_sigma(e, b);
        tree(e, c);
        n_step_tree(e, d);
        largest = std::max(largest, largest_difference(a, b));
        largest = std::max(largest, largest_difference(c, d));
    }
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 1e-12);
}

/// verifies that each call is a whole episode: the traces start at 0, and an episode cut short does not learn its last link
TEST_CASE("off-policy traces: each call is a whole episode, and an episode cut short", "[off_policy_traces]")
{
    const episode e {{state(1, "x"), action("a")}, {state(2, "y"), action("b")}, {state(4, "x"), action("b")}};
    const std::map<std::string, double> pi = {{"xa", 0.8}, {"xb", 0.2}, {"ya", 0.4}, {"yb", 0.6}};
    auto target = [pi](const state & s, const action & a) { return pi.at(s.trait() + a.trait()); };
    auto probabilities = [pi](const state & s) {
        return distribution {{action("a"), pi.at(s.trait() + "a")}, {action("b"), pi.at(s.trait() + "b")}};
    };
    auto b = [](const state &, const action &) { return 0.5; };
    // cut short after two steps: (x, b) is only the next link, so the first two steps alone are learnt, the second
    // bootstrapping from V(x) or V̄(x): the values of the hand-worked test after t = 1
    relearn::off_policy_td_lambda<state,action> td(0.5, 0.5, 0.5, target, b);
    td(e, false);
    REQUIRE(td.value(state("x")) == Approx(1.376));
    REQUIRE(td.value(state("y")) == Approx(1.44));
    relearn::off_policy_expected_sarsa_lambda<state,action> sampled(0.5, 0.5, 0.5, probabilities, b);
    relearn::tree_backup_lambda<state,action> tree(0.5, 0.5, 0.5, probabilities);
    policy q, p;
    sampled(e, q, false);
    tree(e, p, false);
    REQUIRE(q.value(state("x"), action("a")) == Approx(0.83));
    REQUIRE(q.value(state("y"), action("b")) == Approx(1.1));
    REQUIRE(q.value(state("x"), action("b")) == 0);
    REQUIRE(p.value(state("x"), action("a")) == Approx(0.665));
    REQUIRE(p.value(state("y"), action("b")) == Approx(1.1));
    // the next call is a new episode: its traces start at 0, whatever the last one left
    sampled(episode {{state(3, "y"), action("a")}}, q);
    REQUIRE(sampled.traces.entries().size() == 1);
    REQUIRE(sampled.traces.trace(state("y"), action("a")) == Approx(1));
    REQUIRE(q.value(state("x"), action("a")) == Approx(0.83));
    td(episode {{state(3, "y"), action("a")}});
    REQUIRE(td.traces.entries().size() == 1);
    REQUIRE(td.value(state("x")) == Approx(1.376));
    // an empty episode learns nothing
    tree(episode {}, p);
    REQUIRE(p.value(state("x"), action("a")) == Approx(0.665));
}

/// verifies control from random play: Tree-Backup(λ) and importance sampling with a greedy π learn q* in the corridor
TEST_CASE("off-policy traces: learn q* in the corridor from random play", "[off_policy_traces][convergence]")
{
    auto exact = corridor::model(0.9);
    auto v_star = exact.optimal_values();
    using S = corridor_types::state;
    using A = corridor_types::action;
    // greedy π, 20000 episodes. Over 10 seeds, with libstdc++ and with libc++, the worst errors were 0.022 to 0.072
    // (tree backup, λ = 0.9) and 0.024 to 0.076 (importance sampling, λ = 0.5)
    relearn::tree_backup_lambda<S,A> tree(0.1, 0.9, 0.9);
    auto learnt = learn_corridor(tree, 1);
    INFO("tree backup: " << worst_error(learnt, exact, v_star));
    REQUIRE(worst_error(learnt, exact, v_star) < 0.2);
    relearn::off_policy_expected_sarsa_lambda<S,A> sampled(0.1, 0.9, 0.5, nullptr, [](const S &, const A &) { return 0.5; });
    learnt = learn_corridor(sampled, 1);
    INFO("importance sampling: " << worst_error(learnt, exact, v_star));
    REQUIRE(worst_error(learnt, exact, v_star) < 0.2);
}

/// verifies prediction from random play: off-policy TD(λ) learns the values of a policy on FrozenLake
TEST_CASE("off_policy_td_lambda: learns the values of a policy on FrozenLake from random play", "[off_policy_traces][off_policy_td][frozen_lake]")
{
    // over 10 seeds, with libstdc++ and with libc++, the mean RMS error was 0.022 to 0.047 (λ = 0) and 0.037 to 0.063 (λ = 0.5)
    const double zero = lake_prediction_error(0.0, 1);
    const double half = lake_prediction_error(0.5, 1);
    INFO("λ = 0: " << zero << ", λ = 0.5: " << half);
    REQUIRE(zero < 0.08);
    REQUIRE(half < 0.1);
}

/// verifies the constructors, and brace initialisation
TEST_CASE("off-policy traces: constructors and brace initialisation", "[off_policy_traces]")
{
    relearn::off_policy_td_lambda<state,action> td;
    REQUIRE(td.alpha == Approx(0.9));
    REQUIRE(td.gamma == Approx(0.9));
    REQUIRE(td.lambda == Approx(0.9));
    REQUIRE(!td.offline);
    REQUIRE(!td.target);
    REQUIRE(!td.behaviour);
    relearn::off_policy_expected_sarsa_lambda<state,action> sampled;
    REQUIRE(sampled.lambda == Approx(0.9));
    REQUIRE(!sampled.probabilities);
    REQUIRE(!sampled.behaviour);
    relearn::tree_backup_lambda<state,action> tree;
    REQUIRE(tree.lambda == Approx(0.9));
    REQUIRE(!tree.probabilities);
    auto braced = relearn::tree_backup_lambda<state,action>{0.1, 1.0, 0.8};
    REQUIRE(braced.alpha == Approx(0.1));
    REQUIRE(braced.gamma == Approx(1));
    REQUIRE(braced.lambda == Approx(0.8));
    auto other = relearn::off_policy_td_lambda<state,action>{0.2, 0.7, 0.3, [](const state &, const action &) { return 1.0; }, [](const state &, const action &) { return 1.0; }};
    REQUIRE(other.alpha == Approx(0.2));
    REQUIRE(other.target);
    REQUIRE(other.behaviour);
}
