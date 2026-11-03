// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/game.hpp"
#include "../examples/games/frozen_lake.hpp"
#include "../examples/games/racetrack.hpp"
#include "../examples/games/branching_task.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of planning with a model (Sutton & Barto, Chapter 8): `sample_outcome`, `trajectory_sampling`,
 * `real_time_dynamic_programming`, `heuristic_search` and `rollout_planner`:
 * - values and choices worked out by hand, on small models
 * - against their definitions: uniform expected updates are sweeps of value iteration on the action values, heuristic
 *   search d moves deep is d sweeps of value iteration, and the rollouts' estimates are the values of the rollout policy
 * - convergence: the optimal values on FrozenLake, RTDP's optimal policy on the racetrack with its values never below
 *   the optimal ones, Figure 8.8's early and late winners, and planning from rough values on the racetrack
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using model_type = relearn::finite_mdp<state,action>;
using policy  = relearn::policy<state,action>;

/**
 * @brief a small model: x and y, and the terminal state end
 * - in x, "a" leads to y (reward 1) and "b" to end (reward 5)
 * - in y, "a" leads to end (reward 2) with 0.5, or back to x (reward 0) with 0.5; "b" leads to end (reward 3)
 */
model_type small_model()
{
    model_type m;
    m.states = {state("x"), state("y"), state("end")};
    m.actions = [](const state & s) {
        return s.trait() == "end" ? std::vector<action>{} : std::vector<action>{action("a"), action("b")};
    };
    m.outcomes = [](const state & s, const action & a) {
        if (s.trait() == "x") {
            return a.trait() == "a" ? std::vector<model_type::outcome>{{state("y"), 1.0, 1.0}}
                                    : std::vector<model_type::outcome>{{state("end"), 5.0, 1.0}};
        }
        return a.trait() == "a" ? std::vector<model_type::outcome>{{state("end"), 2.0, 0.5}, {state("x"), 0.0, 0.5}}
                                : std::vector<model_type::outcome>{{state("end"), 3.0, 1.0}};
    };
    return m;
}

/**
 * @brief a corridor of costs: 0 -> 1 -> 2 -> end, each move costing -1; in every cell "go" moves on, and
 *        "wait" stays put (also -1)
 */
model_type corridor_model()
{
    model_type m;
    m.states = {state("0"), state("1"), state("2"), state("end")};
    m.actions = [](const state & s) {
        return s.trait() == "end" ? std::vector<action>{} : std::vector<action>{action("wait"), action("go")};
    };
    m.outcomes = [](const state & s, const action & a) {
        const std::map<std::string, std::string> next = {{"0", "1"}, {"1", "2"}, {"2", "end"}};
        return std::vector<model_type::outcome>{{a.trait() == "go" ? state(next.at(s.trait())) : s, -1.0, 1.0}};
    };
    return m;
}

using lake_type   = games::frozen_lake;
using lake_types  = games::types<lake_type>;
using lake_dp     = relearn::dynamic_programming<lake_types::state, lake_types::action>;
using race_type   = games::racetrack;
using race_types  = games::types<race_type>;
using race_dp     = relearn::dynamic_programming<race_types::state, race_types::action>;
using task_type   = games::branching_task;
using task_types  = games::types<task_type>;

/**
 * @brief the mean value of the racetrack's start line under a policy, worked out exactly
 * @param choose gives the action of the policy in a state; it is asked only in the states the policy can reach
 * @return minus the expected number of steps to the finish, or minus infinity if, from some state the policy reaches,
 *         it can never reach the finish line
 */
double start_value(const race_type & track,
                   const race_type::model_type & model,
                   const std::function<race_types::action(const race_types::state &)> & choose)
{
    race_dp::policy_type pi;
    std::map<unsigned int, std::vector<unsigned int>> before;
    std::vector<race_types::state> todo;
    for (auto s : track.starts()) {
        todo.push_back(race_types::state(s));
    }
    while (!todo.empty()) {
        auto s = todo.back();
        todo.pop_back();
        if (model.actions(s).empty() || pi.count(s)) {
            continue;
        }
        auto a = choose(s);
        pi.emplace(s, a);
        for (const auto & o : model.outcomes(s, a)) {
            before[o.next.trait()].push_back(s.trait());
            todo.push_back(o.next);
        }
    }
    // the states the finish line can be reached from, backwards from it
    std::map<unsigned int, bool> finishes = {{track.finished(), true}};
    std::vector<unsigned int> back = {track.finished()};
    while (!back.empty()) {
        auto s = back.back();
        back.pop_back();
        for (auto p : before[s]) {
            if (!finishes[p]) {
                finishes[p] = true;
                back.push_back(p);
            }
        }
    }
    for (const auto & entry : pi) {
        if (!finishes[entry.first.trait()]) {
            return -std::numeric_limits<double>::infinity();
        }
    }
    race_dp exact(1.0, 1e-9);
    exact.max_sweeps = 5000;
    exact.evaluate(model, pi);
    double v = 0;
    for (auto s : track.starts()) {
        v += exact.value(race_types::state(s)) / static_cast<double>(track.starts().size());
    }
    return v;
}

/// @return the first action of @param s whose value by @param q is within 1e-9 of the highest
template <class model, class state_type, class values>
auto first_best(const model & m,
                const state_type & s,
                const values & q)
{
    auto actions = m.actions(s);
    double top = q(s, actions[0]);
    for (const auto & a : actions) {
        top = std::max(top, q(s, a));
    }
    std::size_t b = 0;
    while (q(s, actions[b]) < top - 1e-9) {
        b++;
    }
    return actions[b];
}

/**
 * @brief Figure 8.8: the value of the start state under the greedy policy, after a number of expected updates
 * @param branches is b, @param tasks the number of tasks (seeds seed * 1000 + k), of 1000 states each
 * @param how is the distribution of the updates, @param updates their number
 * @return the value, averaged over the tasks
 */
double figure_8_8(unsigned int branches,
                  unsigned int tasks,
                  unsigned int seed,
                  relearn::update_distribution how,
                  std::size_t updates)
{
    double total = 0;
    for (unsigned int k = 0; k < tasks; k++) {
        task_type task(1000, branches, seed * 1000 + k);
        auto model = task.model();
        relearn::trajectory_sampling<task_types::state, task_types::action> sampling(1.0, how, 0.1);
        task_types::policy q;
        std::mt19937 prng(seed * 1000 + k + 500);
        sampling.plan(model, q, task_types::state(task.start()), updates, prng);
        relearn::dynamic_programming<task_types::state, task_types::action>::policy_type greedy;
        for (const auto & s : model.states) {
            if (!model.actions(s).empty()) {
                const bool first = q.value(s, task_types::action(0u)) >= q.value(s, task_types::action(1u));
                greedy.emplace(s, task_types::action(first ? 0u : 1u));
            }
        }
        relearn::dynamic_programming<task_types::state, task_types::action> exact(1.0, 1e-9);
        exact.evaluate(model, greedy);
        total += exact.value(task_types::state(task.start())) / tasks;
    }
    return total;
}
} // namespace

/// verifies that outcomes are drawn by their probabilities
TEST_CASE("sample_outcome: outcomes drawn by their probabilities", "[planning]")
{
    // y with "a": end with 0.5, x with 0.5
    auto m = small_model();
    std::mt19937 prng(1);
    int ends = 0;
    const int n = 20000;
    for (int i = 0; i < n; i++) {
        ends += relearn::sample_outcome(m, state("y"), action("a"), prng).next == state("end") ? 1 : 0;
    }
    // within 4 standard deviations, sqrt(n / 4) = 70.7
    REQUIRE(std::fabs(ends - n / 2.0) < 4 * 70.7);
    // a sure outcome is always drawn
    REQUIRE(relearn::sample_outcome(m, state("x"), action("b"), prng).reward == 5);
}

/// verifies expected updates worked out by hand, the uniform order, and on-policy trajectories
TEST_CASE("trajectory_sampling: expected updates worked out by hand", "[planning][trajectory_sampling]")
{
    auto m = small_model();
    std::mt19937 prng(2);
    // γ = 0.5, Q = 0 to start with: uniform updates go (x, a), (x, b), (y, a), (y, b), then (x, a) again
    // (x, a) = 1 + 0.5 * max Q(y) = 1;  (x, b) = 5;  (y, a) = 0.5 * 2 + 0.5 * (0 + 0.5 * max Q(x) = 2.5) = 2.25;
    // (y, b) = 3;  (x, a) = 1 + 0.5 * 3 = 2.5
    relearn::trajectory_sampling<state,action> uniform(0.5, relearn::update_distribution::uniform);
    policy q;
    uniform.plan(m, q, state("x"), 4, prng);
    REQUIRE(q.value(state("x"), action("a")) == Approx(1));
    REQUIRE(q.value(state("x"), action("b")) == Approx(5));
    REQUIRE(q.value(state("y"), action("a")) == Approx(2.25));
    REQUIRE(q.value(state("y"), action("b")) == Approx(3));
    REQUIRE(uniform.expected_value(m, q, state("x"), action("a")) == Approx(2.5));
    uniform.plan(m, q, state("x"), 1, prng);
    REQUIRE(q.value(state("x"), action("a")) == Approx(2.5));
    REQUIRE(uniform.updates() == 5);
    // on-policy, greedy (ε = 0), in the corridor of costs: every pair starts at 0, so ties are broken at random, but the
    // trajectory always starts in 0, and each update is of the pair taken in the state the trajectory has reached
    auto corridor = corridor_model();
    relearn::trajectory_sampling<state,action> on_policy(1.0, relearn::update_distribution::on_policy, 0.0);
    policy c;
    on_policy.plan(corridor, c, state("0"), 1, prng);
    // the first update is of a pair of state 0, and is worth -1 (both actions lead to a state worth 0 yet)
    const double wait = c.value(state("0"), action("wait"));
    const double go = c.value(state("0"), action("go"));
    REQUIRE(std::min(wait, go) == Approx(-1));
    REQUIRE(std::max(wait, go) == 0);
    REQUIRE(c.value(state("1"), action("go")) == 0);
    // given enough updates, the greedy trajectories go straight to the end: Q(0, go) = -3
    on_policy.plan(corridor, c, state("0"), 200, prng);
    REQUIRE(c.value(state("0"), action("go")) == Approx(-3));
    REQUIRE(c.value(state("2"), action("go")) == Approx(-1));
}

/// verifies the on-policy trajectories: ε, the ties, the start state, and a trajectory carried on from one call to the next
TEST_CASE("trajectory_sampling: on-policy trajectories, worked out by hand", "[planning][trajectory_sampling]")
{
    auto m = corridor_model();
    std::mt19937 prng(11);
    // going is worth 0 and waiting -10 everywhere: greedy (ε = 0) never waits, so Q(0, wait) stays -10
    policy start;
    for (const auto & s : {"0", "1", "2"}) {
        start.update(state(s), action("go"), 0);
        start.update(state(s), action("wait"), -10);
    }
    relearn::trajectory_sampling<state,action> greedy(1.0, relearn::update_distribution::on_policy, 0.0);
    policy q = start;
    greedy.plan(m, q, state("0"), 30, prng);
    REQUIRE(q.value(state("0"), action("wait")) == -10);
    REQUIRE(q.value(state("0"), action("go")) == Approx(-3));
    // the trajectory goes on from one call to the next: one update of (0, go), then one of (1, go)
    relearn::trajectory_sampling<state,action> stepwise(1.0, relearn::update_distribution::on_policy, 0.0);
    policy p = start;
    stepwise.plan(m, p, state("0"), 1, prng);
    REQUIRE(p.value(state("0"), action("go")) == Approx(-1));
    REQUIRE(p.value(state("1"), action("go")) == 0);
    stepwise.plan(m, p, state("0"), 1, prng);
    REQUIRE(p.value(state("1"), action("go")) == Approx(-1));
    REQUIRE(p.value(state("0"), action("go")) == Approx(-1));
    // a trajectory starts in the start state given: from 1, the first update is of (1, go), and state 0 is never met
    relearn::trajectory_sampling<state,action> from_one(1.0, relearn::update_distribution::on_policy, 0.0);
    policy r = start;
    from_one.plan(m, r, state("1"), 10, prng);
    REQUIRE(r.value(state("1"), action("go")) == Approx(-2));
    REQUIRE(r.value(state("0"), action("go")) == 0);
    // ε = 1 explores at every step: waiting in 0 gets updated too
    relearn::trajectory_sampling<state,action> explore(1.0, relearn::update_distribution::on_policy, 1.0);
    policy x = start;
    explore.plan(m, x, state("0"), 200, prng);
    REQUIRE(x.value(state("0"), action("wait")) > -10);
    // ties are broken at random: with every value 0, the first update of a fresh trajectory is of either pair of 0
    int waits = 0;
    for (int i = 0; i < 200; i++) {
        relearn::trajectory_sampling<state,action> fresh(1.0, relearn::update_distribution::on_policy, 0.0);
        policy f;
        fresh.plan(m, f, state("0"), 1, prng);
        waits += f.value(state("0"), action("wait")) < 0 ? 1 : 0;
    }
    REQUIRE(waits > 50);
    REQUIRE(waits < 150);
}

/// verifies RTDP's trials worked out by hand
TEST_CASE("real_time_dynamic_programming: trials worked out by hand", "[planning][rtdp]")
{
    auto m = corridor_model();
    std::mt19937 prng(3);
    relearn::real_time_dynamic_programming<state,action> rtdp(1.0);
    // every value 0: both actions are worth -1 everywhere, so the trial moves on (go) or waits, at random, until it ends;
    // every state it meets is updated to -1 the first time, and after a wait in s, again to -1 + V(s)
    const auto steps = rtdp.trial(m, state("0"), prng);
    REQUIRE(steps == rtdp.updates());
    REQUIRE(steps >= 3);
    REQUIRE(rtdp.value(state("2")) <= -1);
    REQUIRE(rtdp.value(state("end")) == 0);
    // V(s) is the q-value of the best action: with V(1) = -1 and V(0) = -1, q(0, go) = -2 and q(0, wait) = -2
    relearn::real_time_dynamic_programming<state,action> fresh(1.0);
    fresh.values[state("0")] = -1;
    fresh.values[state("1")] = -1;
    fresh.values[state("2")] = -5;
    REQUIRE(fresh.q_value(m, state("0"), action("go")) == Approx(-2));
    REQUIRE(fresh.q_value(m, state("2"), action("go")) == Approx(-1));
    // in 2, going (-1) beats waiting (-6); in 1, going is -6 and waiting -2
    REQUIRE(fresh.greedy(m, state("2"), prng) == action("go"));
    REQUIRE(fresh.greedy(m, state("1"), prng) == action("wait"));
    // a trial from 2 updates V(2) to -1 and ends; with a step limit of 0 it does nothing
    REQUIRE(fresh.trial(m, state("2"), prng) == 1);
    REQUIRE(fresh.value(state("2")) == Approx(-1));
    REQUIRE(fresh.trial(m, state("0"), prng, 0) == 0);
    // discounted, γ = 0.9: the trials learn the optimal values of the path, V(2) = -1, V(1) = -1 - 0.9 = -1.9 and
    // V(0) = -1 - 0.9 * 1.9 = -2.71 (waiting is worse everywhere: in 1, -1 + 0.9 * -1.9 = -2.71)
    relearn::real_time_dynamic_programming<state,action> discounted(0.9);
    for (int k = 0; k < 50; k++) {
        discounted.trial(m, state("0"), prng);
    }
    REQUIRE(discounted.value(state("2")) == Approx(-1));
    REQUIRE(discounted.value(state("1")) == Approx(-1.9));
    REQUIRE(discounted.value(state("0")) == Approx(-2.71));
    REQUIRE(discounted.q_value(m, state("1"), action("wait")) == Approx(-2.71));
    // ties are broken at random: from all 0, both actions of state 0 are chosen
    relearn::real_time_dynamic_programming<state,action> ties(1.0);
    int go = 0;
    for (int i = 0; i < 200; i++) {
        go += ties.greedy(m, state("0"), prng) == action("go") ? 1 : 0;
    }
    REQUIRE(go > 50);
    REQUIRE(go < 150);
}

/// verifies heuristic search worked out by hand
TEST_CASE("heuristic_search: searches worked out by hand", "[planning][heuristic_search]")
{
    auto m = small_model();
    // γ = 1, the heuristic V(x) = 10, V(y) = 0. Depth 1 from x: a is 1 + V(y) = 1, b is 5: b.
    // Depth 2 from x: a is 1 + V_1(y), V_1(y) = max(0.5 * 2 + 0.5 * V(x) = 6, 3) = 6, so a is 7: a.
    relearn::heuristic_search<state,action> one(1.0, 1);
    one.values[state("x")] = 10;
    one.values[state("y")] = 0;
    REQUIRE(one.choose(m, state("x")) == action("b"));
    REQUIRE(one.q_value(m, state("x"), action("a")) == Approx(1));
    relearn::heuristic_search<state,action> two(1.0, 2);
    two.values = one.values;
    REQUIRE(two.search(m, state("y"), 1) == Approx(6));
    REQUIRE(two.q_value(m, state("x"), action("a")) == Approx(7));
    REQUIRE(two.choose(m, state("x")) == action("a"));
    // depth 0 is the heuristic, and a terminal state is worth 0 at any depth
    REQUIRE(two.search(m, state("x"), 0) == Approx(10));
    REQUIRE(two.search(m, state("end"), 3) == 0);
    REQUIRE(two.value(state("end")) == 0);
    // a tie goes to the first action: all 0 in the corridor, waiting and going are both -1
    relearn::heuristic_search<state,action> flat(1.0, 1);
    REQUIRE(flat.choose(corridor_model(), state("0")) == action("wait"));
}

/// verifies rollouts worked out by hand
TEST_CASE("rollout_planner: rollouts worked out by hand", "[planning][rollouts]")
{
    // in the corridor of costs, a rollout policy that always goes: from 0, going is -3 and waiting -4, exactly
    auto m = corridor_model();
    std::mt19937 prng(4);
    relearn::rollout_planner<state,action> planner(1.0, 5,
        [](const state &, const std::vector<action> &, std::mt19937 &) { return action("go"); });
    REQUIRE(planner.rollout(m, state("0"), action("go"), prng) == Approx(-3));
    REQUIRE(planner.estimate(m, state("0"), action("wait"), prng) == Approx(-4));
    REQUIRE(planner.choose(m, state("0"), prng) == action("go"));
    // discounted, γ = 0.5: -1 - 0.5 - 0.25 = -1.75
    planner.gamma = 0.5;
    REQUIRE(planner.rollout(m, state("0"), action("go"), prng) == Approx(-1.75));
    // a rollout policy that always waits never ends: the episode is cut at max_steps, -1 a step
    relearn::rollout_planner<state,action> stuck(1.0, 1,
        [](const state &, const std::vector<action> &, std::mt19937 &) { return action("wait"); });
    stuck.max_steps = 10;
    REQUIRE(stuck.rollout(m, state("0"), action("go"), prng) == Approx(-10));
    // the default rollout policy picks at random: from 2, going ends at once (-1), waiting costs at least -2
    relearn::rollout_planner<state,action> random_rollouts(1.0, 50);
    REQUIRE(random_rollouts.estimate(m, state("2"), action("go"), prng) == Approx(-1));
    REQUIRE(random_rollouts.estimate(m, state("2"), action("wait"), prng) <= -2);
    REQUIRE(random_rollouts.choose(m, state("2"), prng) == action("go"));
}

/// verifies that uniform expected updates are sweeps of value iteration on the action values, and that they converge to q*
TEST_CASE("trajectory_sampling: uniform updates are sweeps of value iteration on the action values", "[planning][trajectory_sampling]")
{
    lake_type lake;
    auto model = lake.model();
    const double gamma = 0.95;
    // value iteration on the action values, written from its definition: in place, state after state, action after action
    lake_types::policy expected;
    std::size_t pairs = 0;
    auto sweep = [&]() {
        for (const auto & s : model.states) {
            for (const auto & a : model.actions(s)) {
                double q = 0;
                for (const auto & o : model.outcomes(s, a)) {
                    double best = model.actions(o.next).empty() ? 0 : -std::numeric_limits<double>::infinity();
                    for (const auto & b : model.actions(o.next)) {
                        best = std::max(best, expected.value(o.next, b));
                    }
                    q += o.probability * (o.reward + gamma * best);
                }
                expected.update(s, a, q);
                pairs++;
            }
        }
    };
    for (int k = 0; k < 3; k++) {
        sweep();
    }
    relearn::trajectory_sampling<lake_types::state, lake_types::action> uniform(gamma, relearn::update_distribution::uniform);
    lake_types::policy q;
    std::mt19937 prng(5);
    uniform.plan(model, q, lake_types::state(0u), pairs, prng);
    double largest = 0;
    for (const auto & s : model.states) {
        for (const auto & a : model.actions(s)) {
            largest = std::max(largest, std::fabs(q.value(s, a) - expected.value(s, a)));
        }
    }
    REQUIRE(largest < 1e-12);
    // and given enough of them, the optimal values q*, from value iteration
    uniform.plan(model, q, lake_types::state(0u), 1000 * pairs / 3, prng);
    lake_dp optimal(gamma, 1e-12);
    optimal.value_iteration(model);
    largest = 0;
    for (const auto & s : model.states) {
        for (const auto & a : model.actions(s)) {
            largest = std::max(largest, std::fabs(q.value(s, a) - optimal.q_value(model, s, a)));
        }
    }
    REQUIRE(largest < 1e-9);
}

/// verifies that heuristic search d moves deep from V is d sweeps of value iteration (the two-array version) from V
TEST_CASE("heuristic_search: d moves deep is d sweeps of value iteration", "[planning][heuristic_search]")
{
    lake_type lake;
    auto model = lake.model();
    std::mt19937 prng(6);
    std::uniform_real_distribution<double> uniform(0, 1);
    double largest = 0;
    for (unsigned int d = 1; d <= 4; d++) {
        // a random heuristic, from which both start
        relearn::heuristic_search<lake_types::state, lake_types::action> search(0.9, d);
        lake_dp sweeps(0.9, 0);
        sweeps.in_place = false;
        sweeps.max_sweeps = d;
        for (const auto & s : model.states) {
            if (!model.actions(s).empty()) {
                const double v = uniform(prng);
                search.values[s] = v;
                sweeps.values[s] = v;
            }
        }
        sweeps.value_iteration(model);
        for (const auto & s : model.states) {
            largest = std::max(largest, std::fabs(search.search(model, s, d) - sweeps.value(s)));
        }
    }
    REQUIRE(largest < 1e-12);
}

/// verifies that the rollouts' estimates are the values of the rollout policy
TEST_CASE("rollout_planner: the estimates are q of the rollout policy", "[planning][rollouts]")
{
    // FrozenLake, γ = 0.95, with random rollouts: q of the random policy, worked out exactly
    lake_type lake;
    auto model = lake.model();
    lake_dp random_values(0.95, 1e-12);
    random_values.evaluate(model, [](const lake_types::state &, const lake_types::action &) { return 0.25; });
    relearn::rollout_planner<lake_types::state, lake_types::action> rollouts(0.95, 4000);
    std::mt19937 prng(7);
    double largest = 0;
    for (unsigned int s : {0u, 6u, 10u, 14u}) {
        for (unsigned int a = 0; a < 4; a++) {
            const double estimate = rollouts.estimate(model, lake_types::state(s), lake_types::action(a), prng);
            largest = std::max(largest, std::fabs(estimate - random_values.q_value(model, lake_types::state(s), lake_types::action(a))));
        }
    }
    // returns are 0 ... 1, so 4000 of them average to within 4 * 0.5 / sqrt(4000) = 0.032
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 0.032);
}

/// verifies the optimal values on FrozenLake with on-policy trajectory sampling, at the start state
TEST_CASE("trajectory_sampling: on-policy updates learn q* of the start state of FrozenLake", "[planning][trajectory_sampling][convergence]")
{
    lake_type lake;
    auto model = lake.model();
    lake_dp optimal(0.95, 1e-12);
    optimal.value_iteration(model);
    relearn::trajectory_sampling<lake_types::state, lake_types::action> on_policy(0.95, relearn::update_distribution::on_policy, 0.5);
    lake_types::policy q;
    std::mt19937 prng(8);
    on_policy.plan(model, q, lake_types::state(0u), 200000, prng);
    double largest = 0;
    for (unsigned int a = 0; a < 4; a++) {
        largest = std::max(largest, std::fabs(q.value(lake_types::state(0u), lake_types::action(a)) -
                                              optimal.q_value(model, lake_types::state(0u), lake_types::action(a))));
    }
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 1e-6);
}

/// verifies RTDP on the racetrack: its values never go below the optimal ones, and its greedy policy becomes optimal
TEST_CASE("real_time_dynamic_programming: the racetrack's optimal policy, from optimistic values", "[planning][rtdp][racetrack]")
{
    race_type track;
    auto model = track.model();
    race_dp optimal(1.0, 1e-9);
    optimal.value_iteration(model);
    const double best = start_value(track, model, [&](const race_types::state & s) {
        return first_best(model, s, [&](const race_types::state & x, const race_types::action & a) { return optimal.q_value(model, x, a); });
    });
    relearn::real_time_dynamic_programming<race_types::state, race_types::action> rtdp(1.0);
    std::mt19937 prng(9);
    std::uniform_int_distribution<std::size_t> pick(0, track.starts().size() - 1);
    for (int k = 0; k < 5000; k++) {
        rtdp.trial(model, race_types::state(track.starts()[pick(prng)]), prng);
    }
    // 0 is optimistic (every value of the optimal policy is below it), and an update keeps an optimistic value optimistic
    double lowest = std::numeric_limits<double>::infinity();
    for (const auto & v : rtdp.values) {
        lowest = std::min(lowest, v.second - optimal.value(v.first));
    }
    REQUIRE(lowest > -1e-9);
    // only part of the states were ever updated
    std::size_t live = model.states.size() - 1;
    REQUIRE(rtdp.values.size() < live / 2);
    // and its greedy policy is optimal, or nearly. Over 10 seeds, with libstdc++ and with libc++, after 5000 trials
    // it was 0.013 to 0.055 steps longer than the optimal one, and RTDP had updated 2512 to 2523 of the 7575 states
    const double greedy = start_value(track, model, [&](const race_types::state & s) {
        return first_best(model, s, [&](const race_types::state & x, const race_types::action & a) { return rtdp.q_value(model, x, a); });
    });
    INFO("the greedy policy " << greedy << ", the optimal one " << best);
    REQUIRE(greedy > best - 0.2);
}

/// verifies Figure 8.8: on-policy updates win early with one branch, and uniform ones late with ten
TEST_CASE("trajectory_sampling: Figure 8.8, on-policy first, uniform last", "[planning][trajectory_sampling][convergence]")
{
    // 10 tasks of 1000 states. Over 10 seeds, with libstdc++ and with libc++, after 250 updates with b = 1 the on-policy
    // updates were 3.33 to 6.32 and the uniform ones 0.27 to 2.23, on-policy ahead by 1.48 or more; after 20000 with
    // b = 10, the uniform ones were ahead by 0.19 to 0.23
    const double on_early = figure_8_8(1, 10, 1, relearn::update_distribution::on_policy, 250);
    const double uniform_early = figure_8_8(1, 10, 1, relearn::update_distribution::uniform, 250);
    const double on_late = figure_8_8(10, 10, 1, relearn::update_distribution::on_policy, 20000);
    const double uniform_late = figure_8_8(10, 10, 1, relearn::update_distribution::uniform, 20000);
    INFO("b = 1 after 250: on-policy " << on_early << ", uniform " << uniform_early);
    INFO("b = 10 after 20000: on-policy " << on_late << ", uniform " << uniform_late);
    REQUIRE(on_early > uniform_early + 1);
    REQUIRE(uniform_late > on_late + 0.05);
}

/// verifies planning from rough values on the racetrack: searching deeper, and rolling the rough policy out, play better
TEST_CASE("heuristic_search and rollout_planner: from rough values, the racetrack is driven better", "[planning][heuristic_search][rollouts][racetrack]")
{
    race_type track;
    auto model = track.model();
    // a few sweeps of value iteration from 0, the two-array version, and the greedy policy on them (the first best action)
    auto rough_values = [&](unsigned int sweeps) {
        race_dp rough(1.0, 0);
        rough.in_place = false;
        rough.max_sweeps = sweeps;
        rough.value_iteration(model);
        return rough;
    };
    auto plan = [&](const race_dp & rough, unsigned int depth) {
        relearn::heuristic_search<race_types::state, race_types::action> search(1.0, depth);
        search.values = rough.values;
        return start_value(track, model, [&](const race_types::state & s) { return search.choose(model, s); });
    };
    // from 8 sweeps, deterministic: the greedy policy takes 14.75 steps, a search 3 moves deep 11.63, the optimal one 11.55
    const auto eight = rough_values(8);
    const double greedy = plan(eight, 1);
    const double deep = plan(eight, 3);
    INFO("from 8 sweeps: greedy " << greedy << ", search 3 deep " << deep);
    REQUIRE(greedy == Approx(-14.7502).margin(0.0005));
    REQUIRE(deep == Approx(-11.6286).margin(0.0005));
    // from 3 sweeps the greedy policy never finishes: it goes round for ever from some of the states it reaches; rolling
    // it out, 20 times for each action, each rollout cut after 200 steps (so one that never finishes scores -200), drives
    // the track in 25 to 33 steps, over 10 seeds, with libstdc++ and with libc++
    const auto three = rough_values(3);
    const double poor = plan(three, 1);
    auto rough_q = [&](const race_types::state & x, const race_types::action & a) { return three.q_value(model, x, a); };
    relearn::rollout_planner<race_types::state, race_types::action> rollouts(1.0, 20,
        [&](const race_types::state & s, const std::vector<race_types::action> &, std::mt19937 &) { return first_best(model, s, rough_q); });
    rollouts.max_steps = 200;
    std::mt19937 prng(10);
    const double rolled = start_value(track, model, [&](const race_types::state & s) { return rollouts.choose(model, s, prng); });
    INFO("from 3 sweeps: greedy " << poor << ", rollouts " << rolled);
    REQUIRE(std::isinf(poor));
    REQUIRE(rolled > -40);
}

/// verifies the constructors, and brace initialisation
TEST_CASE("planning: constructors and brace initialisation", "[planning]")
{
    relearn::trajectory_sampling<state,action> sampling;
    REQUIRE(sampling.gamma == 1);
    REQUIRE(sampling.epsilon == Approx(0.1));
    REQUIRE(sampling.distribution == relearn::update_distribution::on_policy);
    REQUIRE(sampling.updates() == 0);
    auto braced = relearn::trajectory_sampling<state,action>{0.9, relearn::update_distribution::uniform, 0.2};
    REQUIRE(braced.gamma == Approx(0.9));
    REQUIRE(braced.distribution == relearn::update_distribution::uniform);
    REQUIRE(braced.epsilon == Approx(0.2));
    relearn::real_time_dynamic_programming<state,action> rtdp;
    REQUIRE(rtdp.gamma == 1);
    REQUIRE(rtdp.values.empty());
    REQUIRE(rtdp.updates() == 0);
    relearn::heuristic_search<state,action> search;
    REQUIRE(search.depth == 2);
    auto deep = relearn::heuristic_search<state,action>{0.9, 4};
    REQUIRE(deep.depth == 4);
    relearn::rollout_planner<state,action> rollouts;
    REQUIRE(rollouts.rollouts == 100);
    REQUIRE(rollouts.max_steps == 1000);
    REQUIRE(!rollouts.rollout_policy);
}
