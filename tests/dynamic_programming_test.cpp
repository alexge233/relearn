// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/small_gridworld.hpp"
#include "../examples/games/gamblers_problem.hpp"
#include "solvers.hpp"
#include <cmath>
#include <random>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of `dynamic_programming` (Sutton & Barto, Chapter 4):
 * - policy evaluation, value iteration and policy iteration on a small MDP, worked out by hand
 * - Figure 4.1: the values of the random policy in the 4 x 4 gridworld, sweep by sweep
 * - Example 4.3 and Figure 4.3: the gambler's problem, its values and an optimal policy
 * - policy iteration and value iteration agreeing, and both agreeing with the
 *   independent value iteration and policy evaluation of `solvers.hpp`, on random MDPs
 * There is nothing random in dynamic programming: the random MDPs are seeded.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using model   = relearn::finite_mdp<state,action>;
using solver  = relearn::dynamic_programming<state,action>;
using outcome = model::outcome;

using grid_state  = relearn::state<unsigned int>;
using grid_action = relearn::action<unsigned int>;
using grid_solver = relearn::dynamic_programming<grid_state,grid_action>;

/**
 * @brief a small MDP, to work out by hand: two states, A and B, and the terminal state T
 * @return the model
 *
 *     A --left-->  T, r = 1
 *     A --right--> B, r = 0
 *     B --left-->  A, r = 0
 *     B --right--> T, r = 2  (half the time)
 *               or B, r = 0  (the other half)
 */
model two_states()
{
    model m;
    m.states = {state("A"), state("B"), state("T")};
    m.actions = [](const state & s) {
        if (s == state("T")) {
            return std::vector<action> {};
        }
        return std::vector<action> {action("left"), action("right")};
    };
    m.outcomes = [](const state & s,
                    const action & a) {
        if (s == state("A")) {
            return a == action("left") ? std::vector<outcome> {{state("T"), 1, 1}}
                                       : std::vector<outcome> {{state("B"), 0, 1}};
        }
        return a == action("left") ? std::vector<outcome> {{state("A"), 0, 1}}
                                   : std::vector<outcome> {{state("T"), 2, 0.5},
                                                           {state("B"), 0, 0.5}};
    };
    return m;
}

/// @return π(a | s) = 1/2 for every action: the random policy of two actions
double half(const state &,
            const action &)
{
    return 0.5;
}

/// @return π(a | s) = 1/4 for every action: the random policy of Figure 4.1
double quarter(const grid_state &,
               const grid_action &)
{
    return 0.25;
}

/**
 * @brief the gambler's problem, written from Sutton & Barto's rules, for `solvers.hpp`
 * @param p_heads is the probability that the coin comes up heads
 * @return the model: the state is the capital (0 to 100), the action the stake
 */
solvers::tabular gambler_model(double p_heads)
{
    solvers::tabular m;
    m.states  = 101;
    m.actions = 51;
    m.gamma   = 1;
    m.model = [p_heads](int s, int stake) {
        std::vector<solvers::outcome> out;
        // the game is over at 0 and 100, and a stake must be between 1 and min(s, 100 - s)
        if (s <= 0 || s >= 100 || stake < 1 || stake > std::min(s, 100 - s)) {
            return out;
        }
        // heads wins the stake (reaching 100 is worth 1), tails loses it
        out.push_back({p_heads, s + stake, s + stake >= 100 ? 1.0 : 0.0, s + stake >= 100});
        out.push_back({1 - p_heads, s - stake, 0.0, s - stake == 0});
        return out;
    };
    return m;
}

/**
 * @brief the same random MDP, for relearn and for `solvers.hpp`
 *
 * States 0 ... n-1, and n, the terminal state; every state but the terminal one has
 * the same actions, and each action has two or three outcomes, to random states
 * (the terminal one included), with random rewards and probabilities.
 */
struct random_mdp
{
    /// the model, for relearn
    relearn::finite_mdp<grid_state,grid_action> mdp;
    /// the same model, for `solvers.hpp`
    solvers::tabular tabular;

    /**
     * @param states is n, the number of states that are not terminal
     * @param actions is the number of actions of each of them
     * @param gamma is the discount
     * @param seed seeds the random model
     */
    random_mdp(int states,
               int actions,
               double gamma,
               unsigned int seed)
    {
        // draw the outcomes of every state and action
        std::mt19937 prng(seed);
        std::uniform_int_distribution<int> count(2, 3);
        std::uniform_int_distribution<int> next(0, states);
        std::uniform_real_distribution<double> reward(-1, 1);
        std::uniform_real_distribution<double> weight(0.1, 1);
        using table = std::vector<std::vector<std::vector<solvers::outcome>>>;
        table outcomes(static_cast<std::size_t>(states),
                       std::vector<std::vector<solvers::outcome>>(static_cast<std::size_t>(actions)));
        for (auto & of_state : outcomes) {
            for (auto & of_action : of_state) {
                const int n = count(prng);
                double total = 0;
                for (int i = 0; i < n; i++) {
                    const int to = next(prng);
                    const double w = weight(prng);
                    of_action.push_back({w, to, reward(prng), to == states});
                    total += w;
                }
                // the probabilities add up to 1
                for (auto & o : of_action) {
                    o.probability /= total;
                }
            }
        }
        // for `solvers.hpp`: the terminal state n has no outcomes
        tabular.states  = states + 1;
        tabular.actions = actions;
        tabular.gamma   = gamma;
        tabular.model = [outcomes, states](int s, int a) {
            return s == states ? std::vector<solvers::outcome> {}
                               : outcomes[static_cast<std::size_t>(s)][static_cast<std::size_t>(a)];
        };
        // for relearn: the terminal state n has no actions
        for (int s = 0; s <= states; s++) {
            mdp.states.push_back(grid_state(static_cast<unsigned int>(s)));
        }
        mdp.actions = [states, actions](const grid_state & s) {
            std::vector<grid_action> all;
            for (int a = 0; a < actions && static_cast<int>(s.trait()) != states; a++) {
                all.push_back(grid_action(static_cast<unsigned int>(a)));
            }
            return all;
        };
        mdp.outcomes = [outcomes](const grid_state & s,
                                  const grid_action & a) {
            std::vector<relearn::finite_mdp<grid_state,grid_action>::outcome> out;
            for (const auto & o : outcomes[s.trait()][a.trait()]) {
                out.push_back({grid_state(static_cast<unsigned int>(o.next)), o.reward, o.probability});
            }
            return out;
        };
    }
};

/// @return the values of the 16 cells, rounded to one decimal as in Figure 4.1 (halves upwards: -1.75 is -1.7)
std::vector<double> rounded(const grid_solver & dp)
{
    std::vector<double> v;
    for (unsigned int s = 0; s < 16; s++) {
        v.push_back(std::floor(dp.value(grid_state(s)) * 10 + 0.5) / 10);
    }
    return v;
}
} // namespace

/// verifies iterative policy evaluation, sweep by sweep, in place and with two arrays
TEST_CASE("dynamic programming: policy evaluation by hand", "[dynamic_programming]")
{
    // the random policy (1/2 each), γ = 0.5, every value starting at 0:
    // in place, the first sweep updates A, then B with the new value of A
    // - V(A) = 1/2 * (1 + 0.5 * V(T)) + 1/2 * (0 + 0.5 * V(B)) = 0.5 + 0.25 * 0 = 0.5
    // - V(B) = 1/2 * (0 + 0.5 * V(A)) + 1/2 * (1/2 * 2 + 1/2 * 0.5 * V(B)) = 0.125 + 0.5 = 0.625
    solver dp(0.5, 1e-12);
    dp.max_sweeps = 1;
    dp.evaluate(two_states(), half);
    REQUIRE(dp.value(state("A")) == Approx(0.5));
    REQUIRE(dp.value(state("B")) == Approx(0.625));
    REQUIRE(dp.value(state("T")) == Approx(0));
    REQUIRE(dp.sweeps() == 1);
    // the two-array version updates B with the old value of A, 0: V(B) = 0 + 0.5 = 0.5
    solver two(0.5, 1e-12);
    two.max_sweeps = 1;
    two.in_place = false;
    two.evaluate(two_states(), half);
    REQUIRE(two.value(state("A")) == Approx(0.5));
    REQUIRE(two.value(state("B")) == Approx(0.5));
    // both converge to the solution of V(A) = 0.5 + 0.25 V(B) and V(B) = 0.25 V(A) + 0.5 + 0.125 V(B):
    // V(A) = 9/13 and V(B) = 10/13
    for (bool in_place : {true, false}) {
        solver exact(0.5, 1e-12);
        exact.in_place = in_place;
        exact.evaluate(two_states(), half);
        REQUIRE(exact.value(state("A")) == Approx(9.0 / 13));
        REQUIRE(exact.value(state("B")) == Approx(10.0 / 13));
        REQUIRE(exact.sweeps() > 1);
        REQUIRE(exact.sweeps() < 100);
    }
}

/// verifies value iteration, sweep by sweep, and the greedy policy it ends with
TEST_CASE("dynamic programming: value iteration by hand", "[dynamic_programming]")
{
    // γ = 0.5, in place, every value starting at 0:
    // - sweep 1: V(A) = max(1, 0.5 * 0) = 1,      V(B) = max(0.5 * 1, 1 + 0.25 * 0) = 1
    // - sweep 2: V(A) = max(1, 0.5 * 1) = 1,      V(B) = max(0.5 * 1, 1 + 0.25 * 1) = 1.25
    // - sweep 3: V(A) = max(1, 0.5 * 1.25) = 1,   V(B) = 1 + 0.25 * 1.25 = 1.3125
    // and V(B) = 1 + 0.25 V(B) settles at 4/3: left in A, right in B
    std::vector<double> b = {1, 1.25, 1.3125};
    for (std::size_t k = 1; k <= 3; k++) {
        solver dp(0.5, 1e-12);
        dp.max_sweeps = k;
        dp.value_iteration(two_states());
        REQUIRE(dp.value(state("A")) == Approx(1));
        REQUIRE(dp.value(state("B")) == Approx(b[k - 1]));
    }
    solver dp(0.5, 1e-12);
    auto pi = dp.value_iteration(two_states());
    REQUIRE(dp.value(state("A")) == Approx(1));
    REQUIRE(dp.value(state("B")) == Approx(4.0 / 3));
    REQUIRE(pi.size() == 2);
    REQUIRE(pi.at(state("A")) == action("left"));
    REQUIRE(pi.at(state("B")) == action("right"));
    // q(s,a), one step ahead: q(A, right) = 0.5 * 4/3 = 2/3, q(B, left) = 0.5 * 1 = 0.5
    auto m = two_states();
    REQUIRE(dp.q_value(m, state("A"), action("right")) == Approx(2.0 / 3));
    REQUIRE(dp.q_value(m, state("B"), action("left")) == Approx(0.5));
    // and written into a relearn policy, whose best actions are the greedy policy
    relearn::policy<state,action> memory;
    dp.action_values(m, memory);
    REQUIRE(memory.value(state("A"), action("left")) == Approx(1));
    REQUIRE(memory.value(state("B"), action("right")) == Approx(4.0 / 3));
    REQUIRE(*memory.best_action(state("A")) == action("left"));
    REQUIRE(*memory.best_action(state("B")) == action("right"));
    REQUIRE(memory.actions(state("T")).empty());
}

/// verifies policy iteration, step by step
TEST_CASE("dynamic programming: policy iteration by hand", "[dynamic_programming]")
{
    // γ = 0.5, starting from the policy that goes round in circles: right in A, left in B.
    // 1. it never earns anything: V(A) = V(B) = 0
    //    improvement: in A, left gives 1 and right 0; in B, left gives 0 and right 1: left, right
    // 2. evaluation: V(A) = 1, V(B) = 1 + 0.25 V(B) = 4/3
    //    improvement: in A, left gives 1 and right 2/3; in B, left gives 0.5 and right 4/3: stable
    solver dp(0.5, 1e-12);
    solver::policy_type start;
    start.emplace(state("A"), action("right"));
    start.emplace(state("B"), action("left"));
    // the first evaluation alone
    dp.evaluate(two_states(), start);
    REQUIRE(dp.value(state("A")) == Approx(0));
    REQUIRE(dp.value(state("B")) == Approx(0));
    auto improved = start;
    REQUIRE_FALSE(dp.improve(two_states(), improved));
    REQUIRE(improved.at(state("A")) == action("left"));
    REQUIRE(improved.at(state("B")) == action("right"));
    // the whole of it
    solver all(0.5, 1e-12);
    auto pi = all.policy_iteration(two_states(), start);
    REQUIRE(all.iterations() == 2);
    REQUIRE(pi.at(state("A")) == action("left"));
    REQUIRE(pi.at(state("B")) == action("right"));
    REQUIRE(all.value(state("A")) == Approx(1));
    REQUIRE(all.value(state("B")) == Approx(4.0 / 3));
    REQUIRE(all.value(state("T")) == Approx(0));
}

/// verifies that equally good actions do not make policy iteration switch between them (Exercise 4.4)
TEST_CASE("dynamic programming: ties keep the current action", "[dynamic_programming]")
{
    // in A, both actions end the game with r = 1. Starting from y, policy iteration keeps y,
    // and is stable after one evaluation; without a policy, the first action, x, is taken
    model m;
    m.states = {state("A"), state("T")};
    m.actions = [](const state & s) {
        return s == state("A") ? std::vector<action> {action("x"), action("y")}
                               : std::vector<action> {};
    };
    m.outcomes = [](const state &,
                    const action &) {
        return std::vector<outcome> {{state("T"), 1, 1}};
    };
    solver dp(0.9, 1e-12);
    solver::policy_type start;
    start.emplace(state("A"), action("y"));
    auto pi = dp.policy_iteration(m, start);
    REQUIRE(dp.iterations() == 1);
    REQUIRE(pi.at(state("A")) == action("y"));
    REQUIRE(dp.greedy(m).at(state("A")) == action("x"));
    REQUIRE(dp.value_iteration(m).at(state("A")) == action("x"));
}

/// verifies Figure 4.1: the values of the random policy in the 4 x 4 gridworld, sweep by sweep
TEST_CASE("dynamic programming: Figure 4.1, the random policy in the 4 x 4 gridworld", "[dynamic_programming][small_gridworld]")
{
    // Figure 4.1 updates every value from the values of the sweep before (two arrays), γ = 1.
    // After one sweep every value is -1. After two, a cell next to a terminal corner,
    // e.g. 1, is -1 + 1/4 * (V(T) + V(1) + V(2) + V(5)) = -1 + 1/4 * (0 - 3) = -1.75 (printed -1.7),
    // and the others -2. After three, V(1) = -1 + 1/4 * (0 - 1.75 - 2 - 2) = -2.4375 (-2.4), and so on
    games::small_gridworld game;
    auto m = game.model();
    const std::vector<std::vector<double>> figure = {
        {0, -1.0, -1.0, -1.0,   -1.0, -1.0, -1.0, -1.0,   -1.0, -1.0, -1.0, -1.0,   -1.0, -1.0, -1.0, 0},
        {0, -1.7, -2.0, -2.0,   -1.7, -2.0, -2.0, -2.0,   -2.0, -2.0, -2.0, -1.7,   -2.0, -2.0, -1.7, 0},
        {0, -2.4, -2.9, -3.0,   -2.4, -2.9, -3.0, -2.9,   -2.9, -3.0, -2.9, -2.4,   -3.0, -2.9, -2.4, 0}};
    for (std::size_t k = 1; k <= 3; k++) {
        grid_solver dp(1.0, 1e-12);
        dp.in_place = false;
        dp.max_sweeps = k;
        dp.evaluate(m, quarter);
        REQUIRE(rounded(dp) == figure[k - 1]);
    }
    // k = 2 and 3 exactly, before rounding
    grid_solver three(1.0, 1e-12);
    three.in_place = false;
    three.max_sweeps = 3;
    three.evaluate(m, quarter);
    REQUIRE(three.value(grid_state(1)) == Approx(-2.4375));
    REQUIRE(three.value(grid_state(2)) == Approx(-2.9375));
    REQUIRE(three.value(grid_state(5)) == Approx(-2.875));
    // k = 10, as printed in the book
    grid_solver ten(1.0, 1e-12);
    ten.in_place = false;
    ten.max_sweeps = 10;
    ten.evaluate(m, quarter);
    REQUIRE(rounded(ten) == std::vector<double> {0, -6.1, -8.4, -9.0,   -6.1, -7.7, -8.4, -8.4,
                                                 -8.4, -8.4, -7.7, -6.1,   -9.0, -8.4, -6.1, 0});
    // k = ∞, in place and with two arrays: the exact values of the random policy
    const std::vector<double> random_values = {  0, -14, -20, -22,
                                               -14, -18, -20, -20,
                                               -20, -20, -18, -14,
                                               -22, -20, -14,   0};
    for (bool in_place : {true, false}) {
        grid_solver dp(1.0, 1e-12);
        dp.in_place = in_place;
        dp.evaluate(m, quarter);
        for (unsigned int s = 0; s < 16; s++) {
            REQUIRE(dp.value(grid_state(s)) == Approx(random_values[s]).margin(1e-9));
        }
        // the greedy policy on these values is optimal: minus the moves to the nearest corner
        auto pi = dp.greedy(m);
        grid_solver best(1.0, 1e-12);
        best.evaluate(m, pi);
        const std::vector<double> optimal = { 0, -1, -2, -3,   -1, -2, -3, -2,
                                             -2, -3, -2, -1,   -3, -2, -1,  0};
        for (unsigned int s = 0; s < 16; s++) {
            REQUIRE(best.value(grid_state(s)) == Approx(optimal[s]).margin(1e-9));
        }
    }
}

/// verifies that policy iteration and value iteration find the same optimal values in the gridworld
TEST_CASE("dynamic programming: the 4 x 4 gridworld, policy iteration and value iteration", "[dynamic_programming][small_gridworld]")
{
    games::small_gridworld game;
    auto m = game.model();
    const std::vector<double> optimal = { 0, -1, -2, -3,   -1, -2, -3, -2,
                                         -2, -3, -2, -1,   -3, -2, -1,  0};
    // value iteration, γ = 1: the values settle after 4 sweeps (3 to settle, 1 to see nothing change)
    grid_solver vi(1.0, 1e-12);
    auto by_values = vi.value_iteration(m);
    REQUIRE(vi.sweeps() == 4);
    // policy iteration, γ = 1, starting from the greedy policy of the random policy's values (Figure 4.1)
    grid_solver random_values(1.0, 1e-12);
    random_values.evaluate(m, quarter);
    grid_solver pi(1.0, 1e-12);
    auto by_policy = pi.policy_iteration(m, random_values.greedy(m));
    REQUIRE(pi.iterations() == 1);
    REQUIRE(pi.sweeps() == 4);
    // each evaluation starts from the values already there: once more, from the optimal policy and
    // the optimal values, a single sweep sees that nothing changes
    pi.policy_iteration(m, by_policy);
    REQUIRE(pi.iterations() == 1);
    REQUIRE(pi.sweeps() == 1);
    for (unsigned int s = 0; s < 16; s++) {
        REQUIRE(vi.value(grid_state(s)) == Approx(optimal[s]).margin(1e-9));
        REQUIRE(pi.value(grid_state(s)) == Approx(optimal[s]).margin(1e-9));
    }
    // policy iteration from the policy that always goes up, which never ends an episode from the top row:
    // with γ = 1 its values fall for ever, so each evaluation stops at `max_sweeps`, and it still ends optimal
    grid_solver up(1.0, 1e-12);
    up.max_sweeps = 1000;
    grid_solver::policy_type always_up;
    for (unsigned int s = 1; s < 15; s++) {
        always_up.emplace(grid_state(s), grid_action(games::up));
    }
    up.policy_iteration(m, always_up);
    REQUIRE(up.sweeps() > 1000);
    for (unsigned int s = 0; s < 16; s++) {
        REQUIRE(up.value(grid_state(s)) == Approx(optimal[s]).margin(1e-9));
    }
    // with γ = 0.9 every policy ends (in value), and both agree again
    grid_solver vi_discounted(0.9, 1e-12);
    grid_solver pi_discounted(0.9, 1e-12);
    vi_discounted.value_iteration(m);
    pi_discounted.policy_iteration(m);
    for (unsigned int s = 0; s < 16; s++) {
        REQUIRE(vi_discounted.value(grid_state(s)) == Approx(pi_discounted.value(grid_state(s))).margin(1e-9));
    }
    // two cells away: -1 - 0.9 = -1.9
    REQUIRE(vi_discounted.value(grid_state(2)) == Approx(-1.9));
}

/// verifies Example 4.3: the values of the gambler's problem, and an optimal policy (Figure 4.3)
TEST_CASE("dynamic programming: Example 4.3, the gambler's problem", "[dynamic_programming][gambler]")
{
    games::gamblers_problem game(0.4);
    auto m = game.model();
    grid_solver dp(1.0, 1e-12);
    auto pi = dp.value_iteration(m);
    // the value of a capital is the probability of reaching 100: V(25) = p^2, V(50) = p, V(75) = p + (1 - p) p
    REQUIRE(dp.value(grid_state(25)) == Approx(0.16));
    REQUIRE(dp.value(grid_state(50)) == Approx(0.4));
    REQUIRE(dp.value(grid_state(75)) == Approx(0.64));
    // all the values, against the independent value iteration of `solvers.hpp`
    auto exact = gambler_model(0.4);
    auto v = exact.optimal_values();
    for (unsigned int s = 0; s <= 100; s++) {
        REQUIRE(dp.value(grid_state(s)) == Approx(v[s]).margin(1e-9));
    }
    // the policy is an optimal one: following it is worth the optimal values
    std::vector<int> stakes(101, 0);
    for (unsigned int s = 1; s < 100; s++) {
        stakes[s] = static_cast<int>(pi.at(grid_state(s)).trait());
    }
    auto followed = exact.policy_values(stakes);
    for (unsigned int s = 1; s < 100; s++) {
        REQUIRE(followed[s] == Approx(v[s]).margin(1e-9));
    }
    // many stakes are equally good; taking the smallest of them gives the policy of Figure 4.3:
    // all in at 25, 50 and 75, and in between, stakes that rise from 1 to 12 and fall back to 1
    for (int s = 1; s < 100; s++) {
        // e.g. capitals 26 to 37 stake 1 to 12, and 38 to 49 stake 12 to 1
        const int in_quarter = s % 25;
        int expected = in_quarter <= 12 ? in_quarter : 25 - in_quarter;
        if (in_quarter == 0) {
            expected = std::min(s, 100 - s);
        }
        REQUIRE(stakes[static_cast<std::size_t>(s)] == expected);
    }
    // policy iteration finds the same values (with other stakes among the equally good ones)
    grid_solver by_policy(1.0, 1e-12);
    by_policy.policy_iteration(m);
    for (unsigned int s = 0; s <= 100; s++) {
        REQUIRE(by_policy.value(grid_state(s)) == Approx(v[s]).margin(1e-9));
    }
}

/// verifies policy evaluation, value iteration and policy iteration against `solvers.hpp`, on random MDPs
TEST_CASE("dynamic programming: random MDPs, against solvers.hpp", "[dynamic_programming]")
{
    // 20 random MDPs of 8 states (and a terminal one), 3 actions each, γ = 0.9
    for (unsigned int seed = 1; seed <= 20; seed++) {
        random_mdp r(8, 3, 0.9, seed);
        auto v = r.tabular.optimal_values();
        // value iteration, in place and with two arrays, and policy iteration
        for (bool in_place : {true, false}) {
            grid_solver vi(0.9, 1e-12);
            vi.in_place = in_place;
            auto pi = vi.value_iteration(r.mdp);
            grid_solver by_policy(0.9, 1e-12);
            by_policy.in_place = in_place;
            auto pi_policy = by_policy.policy_iteration(r.mdp);
            for (unsigned int s = 0; s <= 8; s++) {
                REQUIRE(vi.value(grid_state(s)) == Approx(v[s]).margin(1e-9));
                REQUIRE(by_policy.value(grid_state(s)) == Approx(v[s]).margin(1e-9));
            }
            // the actions of a random MDP are hardly ever tied: both find the same policy
            for (unsigned int s = 0; s < 8; s++) {
                REQUIRE(pi.at(grid_state(s)) == pi_policy.at(grid_state(s)));
            }
        }
        // the values of a random stochastic policy
        std::vector<std::vector<double>> weights(9, std::vector<double>(3, 0));
        std::mt19937 prng(seed + 100);
        std::uniform_real_distribution<double> weight(0.1, 1);
        for (auto & row : weights) {
            double total = 0;
            for (auto & w : row) {
                w = weight(prng);
                total += w;
            }
            for (auto & w : row) {
                w /= total;
            }
        }
        grid_solver evaluation(0.9, 1e-12);
        evaluation.evaluate(r.mdp, [&weights](const grid_state & s,
                                              const grid_action & a) {
            return weights[s.trait()][a.trait()];
        });
        auto values = r.tabular.policy_values(weights);
        for (unsigned int s = 0; s <= 8; s++) {
            REQUIRE(evaluation.value(grid_state(s)) == Approx(values[s]).margin(1e-9));
        }
    }
}

/// verifies that the solver works with `float` values
TEST_CASE("dynamic programming: float values", "[dynamic_programming][regression]")
{
    // value iteration on the 4 x 4 gridworld, in float
    using float_model  = relearn::finite_mdp<grid_state,grid_action,float>;
    using float_solver = relearn::dynamic_programming<grid_state,grid_action,float>;
    auto m = games::small_gridworld().model();
    float_model fm;
    fm.states  = m.states;
    fm.actions = m.actions;
    fm.outcomes = [&m](const grid_state & s,
                       const grid_action & a) {
        std::vector<float_model::outcome> out;
        for (const auto & o : m.outcomes(s, a)) {
            out.push_back({o.next, static_cast<float>(o.reward), static_cast<float>(o.probability)});
        }
        return out;
    };
    float_solver dp(1.0f, 1e-6f);
    dp.value_iteration(fm);
    REQUIRE(dp.value(grid_state(3)) == Approx(-3.0f));
}

/// verifies the constructor, and brace initialisation
TEST_CASE("dynamic programming: constructor and brace initialisation", "[dynamic_programming]")
{
    solver defaults;
    REQUIRE(defaults.gamma == Approx(0.9));
    REQUIRE(defaults.theta == Approx(1e-9));
    REQUIRE(defaults.ties == Approx(1e-9));
    REQUIRE(defaults.in_place);
    REQUIRE(defaults.max_sweeps == 100000);
    REQUIRE(defaults.values.empty());
    REQUIRE(defaults.sweeps() == 0);
    REQUIRE(defaults.iterations() == 0);
    auto dp = solver{1.0, 1e-6};
    REQUIRE(dp.gamma == Approx(1.0));
    REQUIRE(dp.theta == Approx(1e-6));
}
