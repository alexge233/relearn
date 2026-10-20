// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/access_control.hpp"
#include "solvers.hpp"
#include <cmath>
#include <map>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of the learners for tasks that never end:
 * - `r_learning` (Sutton & Barto, 1st edition, Section 6.7): the updates of Q and of ρ worked
 *   out by hand, including that ρ only moves after a greedy action
 * - `differential_sarsa` (2nd edition, Section 10.3): its updates worked out by hand
 * - the access-control queuing task: its rules, its exact solution by relative value iteration
 *   (`tests/solvers.hpp`), and both learners against that solution (1st edition Example 6.7,
 *   2nd edition Example 10.2)
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;

using game_type = games::access_control;
using types     = games::types<game_type>;

/// @return the index of state @param s in the model of `solvers::access_control`: 4 * free + k, for the priority 2^k
std::size_t index_of(const game_type::state_type & s)
{
    unsigned int k = 0;
    while (k < 3 && (1u << k) != s.priority) {
        k++;
    }
    return 4 * s.free + k;
}

/// @return the state of index @param s in the model of `solvers::access_control`
game_type::state_type state_of(std::size_t s)
{
    return {static_cast<unsigned int>(s / 4), 1u << (s % 4)};
}

/**
 * @brief the greedy policy learnt, one action per state of `solvers::access_control`
 * @param memory is the learnt policy
 * @return accept (1) where its value is the higher, reject (0) elsewhere,
 *         and on a tie (both untried); reject with no free server
 */
std::vector<int> greedy_policy(const types::policy & memory)
{
    std::vector<int> actions(44, 0);
    for (std::size_t s = 4; s < 44; s++) {
        types::state s_t(state_of(s));
        double q_accept = memory.value(s_t, types::action(game_type::accept));
        double q_reject = memory.value(s_t, types::action(game_type::reject));
        actions[s] = q_accept > q_reject ? 1 : 0;
    }
    return actions;
}

/// what one run on the queue learnt and did
struct queue_run
{
    /// the mean of the learner's ρ over the second half of the run
    double rho = 0;
    /// the mean reward per step the agent earned over the second half of the run (exploring)
    double reward = 0;
    /// the number of visits of each state, in the order of `solvers::access_control`
    std::vector<std::size_t> visits = std::vector<std::size_t>(44, 0);
    /// the greedy policy learnt
    std::vector<int> greedy;
};

/**
 * @brief learn the queue with ε-greedy (ε = 0.1), learning after every step
 * @param learner is `r_learning` or `differential_sarsa`
 * @param queue is the queue
 * @param steps is the number of steps of the run
 * @param seed seeds the agent
 * @return what was learnt, and the mean ρ and reward of the second half of the run
 *
 * A task that never ends is played in pieces of 1000 steps: `play_online` carries on where the
 * last piece stopped, since the queue's `reset()` does not start again.
 */
template <class learner_type>
queue_run learn_queue(learner_type & learner,
                      game_type & queue,
                      std::size_t steps,
                      unsigned int seed)
{
    std::mt19937 prng(seed);
    types::policy memory;
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
    };
    queue_run run;
    std::size_t pieces = 0;
    for (std::size_t done = 0; done < steps; done += 1000) {
        auto played = games::play_online(queue, agent, learner, memory, 1000);
        // count the states visited (the last link is the state reached, not a step)
        for (std::size_t i = 0; i + 1 < played.episode.size(); i++) {
            run.visits[index_of(played.episode[i].state.trait())]++;
        }
        // the second half: ρ after each piece, and the reward earned
        if (done >= steps / 2) {
            run.rho += learner.rho;
            run.reward += played.total_reward / 1000;
            pieces++;
        }
    }
    run.rho /= static_cast<double>(pieces);
    run.reward /= static_cast<double>(pieces);
    run.greedy = greedy_policy(memory);
    return run;
}
} // namespace

/// verifies one step of R-Learning, worked out by hand, when the action taken is greedy
TEST_CASE("r_learning: one greedy step updates Q, then ρ", "[r_learning]")
{
    // α = 0.5, β = 0.1, ρ = 1. In s0 only a is known; a earns r = 3 and leads to s1,
    // where the best value is Q(s1, x) = 2:
    // Q(s0, a) = 0 + 0.5 * (3 - 1 + 2 - 0) = 2, now the best of s0, so a is greedy and
    // ρ = 1 + 0.1 * (3 - 1 + 2 - max Q(s0)) = 1 + 0.1 * (3 - 1 + 2 - 2) = 1.2
    relearn::r_learning<state,action> learner(0.5, 0.1, 1.0);
    policy memory;
    memory.update(state("s1"), action("x"), 2);
    learner(episode {{state(3, "s0"), action("a")},
                     {state(0, "s1"), action("x")}}, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(2));
    REQUIRE(learner.rho == Approx(1.2));
    // the last link only gave the next state: it is not learnt
    REQUIRE(memory.value(state("s1"), action("x")) == Approx(2));
}

/// verifies that ρ does not move after a non-greedy (exploring) action
TEST_CASE("r_learning: ρ does not move after a non-greedy action", "[r_learning]")
{
    // the same step, but s0 also has b, worth 10: Q(s0, a) still becomes 2,
    // which is less than 10, so a is not greedy, and ρ stays at 1
    relearn::r_learning<state,action> learner(0.5, 0.1, 1.0);
    policy memory;
    memory.update(state("s1"), action("x"), 2);
    memory.update(state("s0"), action("b"), 10);
    learner(episode {{state(3, "s0"), action("a")},
                     {state(0, "s1"), action("x")}}, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(2));
    REQUIRE(learner.rho == Approx(1));
    REQUIRE(memory.value(state("s0"), action("b")) == Approx(10));
}

/// verifies that a_t is greedy if it is a best action *after* its update, as in Figure 6.16
TEST_CASE("r_learning: the greedy test uses the value just updated", "[r_learning]")
{
    // α = 0.5, β = 0.1, ρ = 1, and Q(s1, x) = 2 again.
    // 1. a is worth 0 and b 1: a is not greedy before the update, but a earns 3,
    //    Q(s0, a) = 0 + 0.5 * (3 - 1 + 2 - 0) = 2 > 1, so it is after: ρ = 1 + 0.1 * (3 - 1 + 2 - 2) = 1.2
    relearn::r_learning<state,action> learner(0.5, 0.1, 1.0);
    policy memory;
    memory.update(state("s1"), action("x"), 2);
    memory.update(state("s0"), action("a"), 0);
    memory.update(state("s0"), action("b"), 1);
    learner(episode {{state(3, "s0"), action("a")},
                     {state(0, "s1"), action("x")}}, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(2));
    REQUIRE(learner.rho == Approx(1.2));
    // 2. a is worth 3 and b 2.5: a is greedy before the update, but a earns 0 this time,
    //    Q(s0, a) = 3 + 0.5 * (0 - 1 + 2 - 3) = 2 < 2.5, so it is not after: ρ stays at 1
    relearn::r_learning<state,action> second(0.5, 0.1, 1.0);
    policy second_memory;
    second_memory.update(state("s1"), action("x"), 2);
    second_memory.update(state("s0"), action("a"), 3);
    second_memory.update(state("s0"), action("b"), 2.5);
    second(episode {{state(0, "s0"), action("a")},
                    {state(0, "s1"), action("x")}}, second_memory);
    REQUIRE(second_memory.value(state("s0"), action("a")) == Approx(2));
    REQUIRE(second.rho == Approx(1));
}

/// verifies that ρ is updated with the values just updated, when the next state is the same state
TEST_CASE("r_learning: a step back to the same state", "[r_learning]")
{
    // α = 0.5, β = 0.5, ρ = 0: a in s0 earns 2 and stays in s0.
    // Q(s0, a) = 0 + 0.5 * (2 - 0 + 0 - 0) = 1, greedy, and then
    // ρ = 0 + 0.5 * (2 - 0 + max Q(s0) - max Q(s0)) = 0 + 0.5 * (2 + 1 - 1) = 1
    relearn::r_learning<state,action> learner(0.5, 0.5);
    policy memory;
    learner(episode {{state(2, "s0"), action("a")},
                     {state(0, "s0"), action("a")}}, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(1));
    REQUIRE(learner.rho == Approx(1));
}

/// verifies the steps of an episode are learnt in order, each one with the ρ the one before left
TEST_CASE("r_learning: several steps, in order", "[r_learning]")
{
    // α = 0.5, β = 0.1, ρ = 0, every value 0 at first: s0 -a-> s1 -b-> s2, earning 4 then 2.
    // 1. Q(s0, a) = 0.5 * (4 - 0 + 0 - 0) = 2, greedy: ρ = 0.1 * (4 - 0 + 0 - 2) = 0.2
    // 2. Q(s1, b) = 0.5 * (2 - 0.2 + 0 - 0) = 0.9, greedy: ρ = 0.2 + 0.1 * (2 - 0.2 + 0 - 0.9) = 0.29
    // the last link (s2) is not learnt, unless the episode ended:
    // 3. then it is, with a next state worth 0: Q(s2, c) = 0.5 * (1 - 0.29 + 0 - 0) = 0.355,
    //    greedy: ρ = 0.29 + 0.1 * (1 - 0.29 + 0 - 0.355) = 0.3255
    auto walk = episode {{state(4, "s0"), action("a")},
                         {state(2, "s1"), action("b")},
                         {state(1, "s2"), action("c")}};
    relearn::r_learning<state,action> learner(0.5, 0.1);
    policy memory;
    learner(walk, memory);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(2));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(0.9));
    REQUIRE(learner.rho == Approx(0.29));
    REQUIRE(memory.actions(state("s2")).empty());

    relearn::r_learning<state,action> ended(0.5, 0.1);
    policy ended_memory;
    ended(walk, ended_memory, true);
    REQUIRE(ended_memory.value(state("s2"), action("c")) == Approx(0.355));
    REQUIRE(ended.rho == Approx(0.3255));
}

/// verifies `legal_actions`: the actions not tried yet count 0 in both maxima
TEST_CASE("r_learning: actions not tried yet count 0 with legal_actions", "[r_learning]")
{
    // α = 0.5, β = 0.1, ρ = 1: a earns 0 in s0 and leads to s1, where only x is known, worth -2.
    // Without `legal_actions`, the best of s1 is -2 and a is the only action of s0:
    //    Q(s0, a) = 0 + 0.5 * (0 - 1 - 2 - 0) = -1.5, greedy: ρ = 1 + 0.1 * (0 - 1 - 2 + 1.5) = 0.85
    // With every state having actions a, b, x and y, the untried ones count 0:
    //    Q(s0, a) = 0 + 0.5 * (0 - 1 + 0 - 0) = -0.5, less than b's 0: not greedy, ρ stays at 1
    auto one_step = episode {{state(0, "s0"), action("a")},
                             {state(0, "s1"), action("x")}};
    relearn::r_learning<state,action> stored(0.5, 0.1, 1.0);
    policy stored_memory;
    stored_memory.update(state("s1"), action("x"), -2);
    stored(one_step, stored_memory);
    REQUIRE(stored_memory.value(state("s0"), action("a")) == Approx(-1.5));
    REQUIRE(stored.rho == Approx(0.85));

    relearn::r_learning<state,action> legal(0.5, 0.1, 1.0);
    legal.legal_actions = [](const state &) {
        return std::vector<action> {action("a"), action("b"), action("x"), action("y")};
    };
    policy legal_memory;
    legal_memory.update(state("s1"), action("x"), -2);
    REQUIRE(legal.best_value(state("s1"), legal_memory) == Approx(0));
    legal(one_step, legal_memory);
    REQUIRE(legal_memory.value(state("s0"), action("a")) == Approx(-0.5));
    REQUIRE(legal.rho == Approx(1));
}

/// verifies one step of differential Sarsa, worked out by hand
TEST_CASE("differential_sarsa: one step updates ρ and Q with the same error", "[differential_sarsa]")
{
    // α = 0.5, β = 0.1, ρ = 1. a earns 3 in s0, and the agent takes b next in s1, worth 2
    // (c, worth 10, is better, but Sarsa uses the action taken):
    // δ = 3 - 1 + Q(s1, b) - Q(s0, a) = 3 - 1 + 2 - 0 = 4
    // ρ = 1 + 0.1 * 4 = 1.4, and Q(s0, a) = 0 + 0.5 * 4 = 2 (the δ worked out with the old ρ)
    relearn::differential_sarsa<state,action> learner(0.5, 0.1, 1.0);
    policy memory;
    memory.update(state("s1"), action("b"), 2);
    memory.update(state("s1"), action("c"), 10);
    learner(episode {{state(3, "s0"), action("a")},
                     {state(0, "s1"), action("b")}}, memory);
    REQUIRE(learner.rho == Approx(1.4));
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(2));

    // ρ learns from every step, exploring or not: here a is far from the best action of s0 (z, worth 100)
    // δ = 3 - 1.4 + 2 - 2 = 1.6, ρ = 1.4 + 0.16 = 1.56, Q(s0, a) = 2 + 0.8 = 2.8
    memory.update(state("s0"), action("z"), 100);
    learner(episode {{state(3, "s0"), action("a")},
                     {state(0, "s1"), action("b")}}, memory);
    REQUIRE(learner.rho == Approx(1.56));
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(2.8));
}

/// verifies that differential Sarsa's last link is learnt only if the episode ended
TEST_CASE("differential_sarsa: the last link", "[differential_sarsa]")
{
    // α = 0.5, β = 0.1, ρ = 0: one link, earning 2.
    // Not ended: nothing is learnt. Ended: δ = 2 - 0 + 0 - 0 = 2, ρ = 0.2, Q = 1
    relearn::differential_sarsa<state,action> learner(0.5, 0.1);
    policy memory;
    learner(episode {{state(2, "s0"), action("a")}}, memory);
    REQUIRE(memory.actions(state("s0")).empty());
    REQUIRE(learner.rho == Approx(0));
    learner(episode {{state(2, "s0"), action("a")}}, memory, true);
    REQUIRE(learner.rho == Approx(0.2));
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(1));
}

/// verifies the constructors, and brace initialisation
TEST_CASE("r_learning: constructors and brace initialisation", "[r_learning][differential_sarsa]")
{
    relearn::r_learning<state,action> r_defaults;
    REQUIRE(r_defaults.alpha == Approx(0.1));
    REQUIRE(r_defaults.beta == Approx(0.01));
    REQUIRE(r_defaults.rho == Approx(0));
    auto r = relearn::r_learning<state,action>{0.2, 0.05, 1.5};
    REQUIRE(r.alpha == Approx(0.2));
    REQUIRE(r.beta == Approx(0.05));
    REQUIRE(r.rho == Approx(1.5));
    relearn::differential_sarsa<state,action> d_defaults;
    REQUIRE(d_defaults.alpha == Approx(0.01));
    REQUIRE(d_defaults.beta == Approx(0.01));
    REQUIRE(d_defaults.rho == Approx(0));
    auto d = relearn::differential_sarsa<state,action>{0.2, 0.05, 1.5};
    REQUIRE(d.alpha == Approx(0.2));
    REQUIRE(d.beta == Approx(0.05));
    REQUIRE(d.rho == Approx(1.5));
}

/// verifies that the learners work with `float` values, though a state's reward is a `double`
TEST_CASE("r_learning: float values", "[r_learning][differential_sarsa][regression]")
{
    // the first test of each learner, in float
    relearn::r_learning<state,action,episode,float> r(0.5f, 0.1f, 1.0f);
    relearn::policy<state,action,float> r_memory;
    r_memory.update(state("s1"), action("x"), 2.0f);
    r(episode {{state(3, "s0"), action("a")},
               {state(0, "s1"), action("x")}}, r_memory);
    REQUIRE(r.rho == Approx(1.2f));
    relearn::differential_sarsa<state,action,episode,float> d(0.5f, 0.1f, 1.0f);
    relearn::policy<state,action,float> d_memory;
    d_memory.update(state("s1"), action("b"), 2.0f);
    d(episode {{state(3, "s0"), action("a")},
               {state(0, "s1"), action("b")}}, d_memory);
    REQUIRE(d.rho == Approx(1.4f));
}

/// verifies the rules of the access-control queue
TEST_CASE("access_control: the rules", "[access_control]")
{
    // servers that never become free (p = 0): every accepted customer keeps its server
    game_type full(3, 0.0, 1);
    auto s = full.reset();
    REQUIRE(s.free == 3);
    // reset() does not start again: it returns the state the queue is in
    REQUIRE(full.reset() == s);
    double paid = 0;
    for (unsigned int i = 0; i < 3; i++) {
        unsigned int priority = full.reset().priority;
        auto out = full.step(game_type::accept);
        REQUIRE(out.reward == Approx(priority));
        REQUIRE(out.next.free == 2 - i);
        REQUIRE_FALSE(out.terminated);
        paid += out.reward;
    }
    REQUIRE(paid > 0);
    // no free server: only reject, and accepting anyway pays nothing
    REQUIRE(full.actions(full.reset()) == std::vector<unsigned int> {game_type::reject});
    auto out = full.step(game_type::accept);
    REQUIRE(out.reward == Approx(0));
    REQUIRE(out.next.free == 0);
    // rejecting pays nothing
    game_type queue(10, 0.06, 2);
    REQUIRE(queue.actions(queue.reset()).size() == 2);
    REQUIRE(queue.step(game_type::reject).reward == Approx(0));
    // servers that always become free (p = 1): even the one just taken is free again
    game_type empty(10, 1.0, 3);
    REQUIRE(empty.step(game_type::accept).next.free == 10);
    // the four priorities are equally likely
    std::map<unsigned int, unsigned int> seen;
    for (unsigned int i = 0; i < 40000; i++) {
        seen[queue.step(game_type::reject).next.priority]++;
    }
    REQUIRE(seen.size() == 4);
    for (auto priority : game_type::priorities()) {
        REQUIRE(seen[priority] / 40000.0 == Approx(0.25).margin(0.01));
    }
}

/// verifies the exact solution: the optimal average reward and the optimal policy, to compare with Figures 6.17 and 10.5
TEST_CASE("access_control: the exact solution", "[access_control][solvers]")
{
    auto mdp = solvers::access_control();
    auto best = mdp.optimal_average();
    // relative value iteration closes in on ρ* from both sides
    REQUIRE(best.high - best.low < 1e-10);
    REQUIRE(best.rho == Approx(2.7476).margin(0.0001));
    // the optimal policy: priorities 8 and 4 are always accepted, 2 with at least 4 free servers,
    // and 1 never (with 10 free servers, rejecting 1 is better by only 0.003)
    for (std::size_t free = 1; free <= 10; free++) {
        REQUIRE(best.policy[4 * free + 3] == 1);
        REQUIRE(best.policy[4 * free + 2] == 1);
        REQUIRE(best.policy[4 * free + 1] == (free >= 4 ? 1 : 0));
        REQUIRE(best.policy[4 * free + 0] == 0);
    }
    // the game works out the same average reward, from its own rules,
    // for the optimal policy and for accepting everyone (2.1814)
    game_type queue;
    auto optimal = [&](const game_type::state_type & s) {
        return static_cast<unsigned int>(best.policy[index_of(s)]);
    };
    auto everyone = [](const game_type::state_type &) {
        return game_type::accept;
    };
    REQUIRE(queue.average_reward(optimal) == Approx(best.rho).margin(1e-9));
    std::vector<int> accept_all(44, 1);
    for (std::size_t s = 0; s < 4; s++) {
        accept_all[s] = 0;
    }
    REQUIRE(mdp.policy_average(accept_all) == Approx(2.1814).margin(0.0001));
    REQUIRE(queue.average_reward(everyone) == Approx(mdp.policy_average(accept_all)).margin(1e-9));
    // under the optimal policy, 9 or 10 servers are free in 56 steps in a million
    auto mu = mdp.stationary(best.policy);
    double many_free = 0;
    for (std::size_t s = 36; s < 44; s++) {
        many_free += mu[s];
    }
    REQUIRE(many_free == Approx(0.000056).margin(0.000001));
    // so a policy that rejects every customer with 10 free servers looks optimal for a long while,
    // but once every server is free it never accepts anyone again: in the long run, it earns 0
    auto trapped = best.policy;
    for (std::size_t s = 40; s < 44; s++) {
        trapped[s] = 0;
    }
    REQUIRE(mdp.policy_average(trapped) == Approx(0).margin(1e-9));
    REQUIRE(queue.average_reward([&](const game_type::state_type & s) {
        return static_cast<unsigned int>(trapped[index_of(s)]);
    }) == Approx(0).margin(1e-9));
    // and playing it gives the same: 1000000 steps of accepting everyone earn 2.1814 per step
    double total = 0;
    for (unsigned int i = 0; i < 1000000; i++) {
        total += queue.step(game_type::accept).reward;
    }
    REQUIRE(total / 1000000 == Approx(2.1814).margin(0.02));
}

/// verifies Example 6.7: R-Learning learns the optimal average reward, whilst exploring
TEST_CASE("r_learning: Example 6.7, the access-control queue", "[r_learning][access_control]")
{
    // the 1st edition's settings: α = 0.1, β = 0.01, ε = 0.1, here for 1000000 steps.
    // Over 10 seeds, with libstdc++ and with libc++:
    // - the mean of ρ over the second half was 2.7427 to 2.7663 (ρ* = 2.7476), whilst the
    //   agent, exploring, earned 2.6007 to 2.6174 per step: ρ learns the greedy policy's reward
    // - the greedy policy learnt, taking the optimal action in the states visited fewer than 1000
    //   times, earned (exactly, from the model) 0.0031 to 0.1393 less than the optimal one.
    //   The greedy policy on its own earned 0.0091 to 0.1416 less in 7 runs, but 2.34 to 2.75 less
    //   in the other 13: with every server free, a state seen only a few times, it rejects most
    //   customers, so once the queue gets there it hardly fills again (have a look at docs/r_learning.md)
    game_type queue(10, 0.06, 1);
    relearn::r_learning<types::state, types::action> learner(0.1, 0.01);
    learner.legal_actions = [&queue](const types::state & s) {
        std::vector<types::action> legal;
        for (const auto & a : queue.actions(s.trait())) {
            legal.push_back(types::action(a));
        }
        return legal;
    };
    auto run = learn_queue(learner, queue, 1000000, 101);
    auto mdp = solvers::access_control();
    auto best = mdp.optimal_average();
    REQUIRE(run.rho == Approx(best.rho).margin(0.05));
    REQUIRE(run.rho > run.reward + 0.1);
    // the states seen often are learnt, the others take the optimal action
    auto learnt = run.greedy;
    for (std::size_t s = 4; s < 44; s++) {
        if (run.visits[s] < 1000) {
            learnt[s] = best.policy[s];
        }
        // priority 8 is always worth accepting, by at least 4: learnt everywhere it was seen often
        else if (s % 4 == 3) {
            REQUIRE(learnt[s] == 1);
        }
    }
    REQUIRE(best.rho - mdp.policy_average(learnt) < 0.2);
}

/// verifies Example 10.2: differential Sarsa learns the average reward of the policy it follows
TEST_CASE("differential_sarsa: Example 10.2, the access-control queue", "[differential_sarsa][access_control]")
{
    // the 2nd edition's settings: α = 0.01, β = 0.01, ε = 0.1, here for 1000000 steps.
    // Over 10 seeds, with libstdc++ and with libc++:
    // - the mean of ρ over the second half was within 0.017 of what the agent earned per step
    //   whilst exploring (2.6289 to 2.6404), and 0.092 to 0.133 below ρ* = 2.7476
    // - the greedy policy learnt earned (exactly, from the model) 0.0014 to 0.0469 less than the optimal one
    game_type queue(10, 0.06, 1);
    relearn::differential_sarsa<types::state, types::action> learner(0.01, 0.01);
    auto run = learn_queue(learner, queue, 1000000, 101);
    auto mdp = solvers::access_control();
    auto best = mdp.optimal_average();
    REQUIRE(run.rho == Approx(run.reward).margin(0.04));
    REQUIRE(run.rho < best.rho - 0.05);
    REQUIRE(best.rho - mdp.policy_average(run.greedy) < 0.08);
    // priority 8 is always worth accepting: learnt everywhere it was seen often
    for (std::size_t free = 1; free <= 10; free++) {
        if (run.visits[4 * free + 3] >= 1000) {
            REQUIRE(run.greedy[4 * free + 3] == 1);
        }
    }
}
