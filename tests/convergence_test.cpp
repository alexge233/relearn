#include "relearn.hpp"
#include <random>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Convergence tests: do the learners converge to the *definition* of Q?
 *
 *     Q(s,a) = E[ r_t + γ r_{t+1} + γ² r_{t+2} + ... | s_t = s, a_t = a ]
 *
 * Episode convention used by relearn: each link (s_t, a_t) carries in
 * `s_t.reward()` the reward r_t received *after* taking a_t in s_t.
 * The last link of an episode is the final decision; the episode ends after it.
 *
 * Each MDP below has an exact Q-value, and that value is first checked against
 * a Monte Carlo average of sampled returns, so the ground truth does not depend
 * on any update rule. Random number generators are seeded: results are reproducible.
 */
// set shortcuts to the state, action, link and episode types
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;

/// the discount rate γ used by every test
static const double discount = 0.9;

/**
 * @brief the discounted return of a whole episode, from its first link:
 *        G = r_0 + γ r_1 + γ² r_2 + ...
 * @param ep is the episode
 * @return the discounted return
 */
double discounted_return(const episode & ep)
{
    double g = 0;
    double k = 1;
    for (const auto & link : ep) {
        // add this link's reward, discounted by how far it is from the start
        g += k * link.state.reward();
        k *= discount;
    }
    return g;
}

/**
 * @struct mdp_a
 * @brief MDP A - stochastic outcome of the final decision
 *
 *   s0 --a--> s1 (r = 0);  s1 --b--> end with r = +1 (p = 0.7) or r = -1 (p = 0.3)
 *
 *   Q(s1,b) = 0.7 - 0.3           = 0.4
 *   Q(s0,a) = 0 + γ * Q(s1,b)     = 0.36
 */
struct mdp_a
{
    /// the exact value of Q(s0,a)
    double q_s0 = discount * 0.4;
    /// the exact value of Q(s1,b)
    double q_s1 = 0.4;

    /// @return the policy π, for Expected SARSA: the only action of state @param s, with probability 1
    relearn::expected_sarsa<state,action>::distribution pi(const state & s) const
    {
        if (s.trait() == "s0") {
            return {{action("a"), 1.0}};
        }
        return {{action("b"), 1.0}};
    }

    /// @return one sampled episode, using @param prng for the random outcome
    episode operator()(std::mt19937 & prng) const
    {
        std::bernoulli_distribution win(0.7);
        return {
            {state(0, "s0"), action("a")},
            {state(win(prng) ? 1 : -1, "s1"), action("b")}
        };
    }
};

/**
 * @struct mdp_b
 * @brief MDP B - stochastic transition and an intermediate reward
 *
 *   s0 --a--> r = 0.5, next state s1 (p = 0.6) or s2 (p = 0.4)
 *   s1 --x--> end with r = +1 (p = 0.8) or r = 0 (p = 0.2)
 *   s2 --y--> end with r = -1
 *
 *   Q(s1,x) = 0.8,  Q(s2,y) = -1
 *   Q(s0,a) = 0.5 + γ * (0.6 * 0.8 + 0.4 * -1) = 0.572
 */
struct mdp_b
{
    /// the exact value of Q(s0,a)
    double q_s0 = 0.5 + discount * (0.6 * 0.8 + 0.4 * -1.0);
    /// the exact value of Q(s1,x)
    double q_s1 = 0.8;
    /// the exact value of Q(s2,y)
    double q_s2 = -1.0;

    /// @return the policy π, for Expected SARSA: the only action of state @param s, with probability 1
    relearn::expected_sarsa<state,action>::distribution pi(const state & s) const
    {
        if (s.trait() == "s0") {
            return {{action("a"), 1.0}};
        }
        if (s.trait() == "s1") {
            return {{action("x"), 1.0}};
        }
        return {{action("y"), 1.0}};
    }

    /// @return one sampled episode, using @param prng for the random outcomes
    episode operator()(std::mt19937 & prng) const
    {
        std::bernoulli_distribution to_s1(0.6);
        std::bernoulli_distribution win(0.8);
        // first the transition from s0, then (in s1) the final reward
        if (to_s1(prng)) {
            return {{state(0.5, "s0"), action("a")},
                    {state(win(prng) ? 1 : 0, "s1"), action("x")}};
        }
        return {{state(0.5, "s0"), action("a")},
                {state(-1, "s2"), action("y")}};
    }
};

/// verifies the exact Q-values of both MDPs by averaging sampled returns (Monte Carlo)
TEST_CASE("ground truth: exact Q matches the Monte Carlo return", "[convergence]")
{
    std::mt19937 prng(1);
    const int n = 200000;
    double g_a = 0;
    double g_b = 0;
    for (int i = 0; i < n; i++) {
        g_a += discounted_return(mdp_a{}(prng));
        g_b += discounted_return(mdp_b{}(prng));
    }
    REQUIRE(g_a / n == Approx(mdp_a{}.q_s0).margin(0.01));
    REQUIRE(g_b / n == Approx(mdp_b{}.q_s0).margin(0.01));
}

/// verifies that on a deterministic chain each step is worth γ times the next one
TEST_CASE("q_learning: deterministic chain gives exact discounted values", "[convergence]")
{
    // s0 -> s1 -> s2, reward 1 for the final decision only
    episode chain = {{state(0, "s0"), action("a")},
                     {state(0, "s1"), action("b")},
                     {state(1, "s2"), action("c")}};
    relearn::policy<state,action> memory;
    relearn::q_learning<state,action> learner;
    // deterministic: no averaging needed, so α = 1
    learner.alpha = 1;
    learner.gamma = discount;
    for (int i = 0; i < 3; i++) {
        learner(chain, memory);
    }
    REQUIRE(memory.value(state("s2"), action("c")) == Approx(1.0));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(discount));
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(discount * discount));
}

/**
 * @brief train a `q_learning` learner with a decaying learning rate α = 1/k
 * @param sample draws one episode of the MDP
 * @param episodes is the number of episodes to train on
 * @return the learnt policy
 */
template <class mdp>
relearn::policy<state,action> train_q_learning(const mdp & sample,
                                               int episodes)
{
    std::mt19937 prng(2);
    relearn::policy<state,action> memory;
    relearn::q_learning<state,action> learner;
    learner.gamma = discount;
    for (int k = 1; k <= episodes; k++) {
        // decay the learning rate as α = 1/k, where k counts episodes (not visits).
        // Each update moves Q(s,a) towards r + γ max Q(s',a'), the current estimate
        // of the next state (bootstrapping), not towards a sampled return.
        // While every pair keeps being visited, this decay meets the conditions
        // Σα = ∞ and Σα² < ∞ under which Q-learning converges
        learner.alpha = 1.0 / k;
        learner(sample(prng), memory);
    }
    return memory;
}

/**
 * @brief train a `q_probabilistic` learner (model-based, no learning rate)
 * @param sample draws one episode of the MDP
 * @param episodes is the number of episodes to train on
 * @return the learnt policy
 */
template <class mdp>
relearn::policy<state,action> train_q_probabilistic(const mdp & sample,
                                                    int episodes)
{
    std::mt19937 prng(3);
    relearn::policy<state,action> memory;
    relearn::q_probabilistic<state,action> learner(discount);
    for (int k = 1; k <= episodes; k++) {
        learner(sample(prng), memory);
    }
    return memory;
}

/**
 * @brief train a TD learner (`sarsa` or `expected_sarsa`) with a decaying learning rate α = 1/k
 * @param learner is the learner to train
 * @param sample draws one episode of the MDP
 * @param episodes is the number of episodes to train on
 * @param seed seeds the random outcomes of the MDP
 * @return the learnt policy
 *
 * Each state of these MDPs has a single action, so the policy followed is the
 * only policy there is: its values q_π are the optimal values q*, and SARSA,
 * Expected SARSA and Q-learning must all reach the same exact Q.
 */
template <class learner_type,
          class mdp>
relearn::policy<state,action> train_td(learner_type & learner,
                                       const mdp & sample,
                                       int episodes,
                                       unsigned int seed)
{
    std::mt19937 prng(seed);
    relearn::policy<state,action> memory;
    learner.gamma = discount;
    for (int k = 1; k <= episodes; k++) {
        // the same decaying learning rate as for Q-learning above
        learner.alpha = 1.0 / k;
        learner(sample(prng), memory);
    }
    return memory;
}

/// verifies that `q_learning` with a decaying α reaches the exact Q of both MDPs
TEST_CASE("q_learning converges to the true Q on stochastic MDPs", "[convergence]")
{
    // α = 1/k decays fast for rarely visited states (s2 is reached in 40% of
    // episodes), so their initial value of 0 washes out slowly: use more episodes
    const int n = 200000;
    SECTION("MDP A: stochastic final reward") {
        auto memory = train_q_learning(mdp_a{}, n);
        REQUIRE(memory.value(state("s1"), action("b")) == Approx(mdp_a{}.q_s1).margin(0.02));
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(mdp_a{}.q_s0).margin(0.02));
    }
    SECTION("MDP B: stochastic transition, intermediate reward") {
        auto memory = train_q_learning(mdp_b{}, n);
        REQUIRE(memory.value(state("s1"), action("x")) == Approx(mdp_b{}.q_s1).margin(0.02));
        REQUIRE(memory.value(state("s2"), action("y")) == Approx(mdp_b{}.q_s2).margin(0.02));
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(mdp_b{}.q_s0).margin(0.02));
    }
}

/// verifies that `q_probabilistic` reaches the exact Q of both MDPs
TEST_CASE("q_probabilistic converges to the true Q on stochastic MDPs", "[convergence]")
{
    const int n = 20000;
    SECTION("MDP A: stochastic final reward") {
        auto memory = train_q_probabilistic(mdp_a{}, n);
        REQUIRE(memory.value(state("s1"), action("b")) == Approx(mdp_a{}.q_s1).margin(0.02));
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(mdp_a{}.q_s0).margin(0.02));
    }
    SECTION("MDP B: stochastic transition, intermediate reward") {
        auto memory = train_q_probabilistic(mdp_b{}, n);
        REQUIRE(memory.value(state("s1"), action("x")) == Approx(mdp_b{}.q_s1).margin(0.02));
        REQUIRE(memory.value(state("s2"), action("y")) == Approx(mdp_b{}.q_s2).margin(0.02));
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(mdp_b{}.q_s0).margin(0.02));
    }
}

/// verifies that `sarsa` with a decaying α reaches the exact Q of both MDPs (Sutton & Barto, Section 6.4)
TEST_CASE("sarsa converges to the true Q on stochastic MDPs", "[convergence][sarsa]")
{
    // the same number of episodes as Q-learning: α = 1/k decays fast for s2.
    // With a single action per state, the action taken next, the expected one and
    // the best one are the same, so SARSA, Expected SARSA and Q-learning make the
    // same updates here: these tests check that each reaches the exact values.
    // What tells them apart is tested in sarsa_test.cpp.
    // Across seeds 1 ... 10 the worst errors were 0.0034 (A) and 0.0155 (B)
    const int n = 200000;
    SECTION("MDP A: stochastic final reward") {
        relearn::sarsa<state,action> learner;
        auto memory = train_td(learner, mdp_a{}, n, 4);
        REQUIRE(memory.value(state("s1"), action("b")) == Approx(mdp_a{}.q_s1).margin(0.02));
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(mdp_a{}.q_s0).margin(0.02));
    }
    SECTION("MDP B: stochastic transition, intermediate reward") {
        relearn::sarsa<state,action> learner;
        auto memory = train_td(learner, mdp_b{}, n, 4);
        REQUIRE(memory.value(state("s1"), action("x")) == Approx(mdp_b{}.q_s1).margin(0.02));
        REQUIRE(memory.value(state("s2"), action("y")) == Approx(mdp_b{}.q_s2).margin(0.02));
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(mdp_b{}.q_s0).margin(0.02));
    }
}

/// verifies that `expected_sarsa`, given the policy π, reaches the exact Q of both MDPs (Sutton & Barto, Section 6.6)
TEST_CASE("expected_sarsa converges to the true Q on stochastic MDPs", "[convergence][expected_sarsa]")
{
    // as for SARSA: across seeds 1 ... 10 the worst errors were 0.0034 (A) and 0.0155 (B)
    const int n = 200000;
    SECTION("MDP A: stochastic final reward") {
        relearn::expected_sarsa<state,action> learner;
        learner.probabilities = [](const state & s) {
            return mdp_a{}.pi(s);
        };
        auto memory = train_td(learner, mdp_a{}, n, 5);
        REQUIRE(memory.value(state("s1"), action("b")) == Approx(mdp_a{}.q_s1).margin(0.02));
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(mdp_a{}.q_s0).margin(0.02));
    }
    SECTION("MDP B: stochastic transition, intermediate reward") {
        relearn::expected_sarsa<state,action> learner;
        learner.probabilities = [](const state & s) {
            return mdp_b{}.pi(s);
        };
        auto memory = train_td(learner, mdp_b{}, n, 5);
        REQUIRE(memory.value(state("s1"), action("x")) == Approx(mdp_b{}.q_s1).margin(0.02));
        REQUIRE(memory.value(state("s2"), action("y")) == Approx(mdp_b{}.q_s2).margin(0.02));
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(mdp_b{}.q_s0).margin(0.02));
    }
}

/**
 * @brief train Double Q-learning with α = 1/n per estimate: each estimate is then the
 *        plain average of its own targets
 * @param sample draws one episode of the MDP
 * @param episodes is the number of episodes to train on
 * @param seed seeds the random outcomes of the MDP and the coin
 * @return the learnt policy: the average of both estimates
 */
template <class mdp>
relearn::policy<state,action> train_double_q(const mdp & sample,
                                             int episodes,
                                             unsigned int seed)
{
    std::mt19937 prng(seed);
    relearn::policy<state,action> memory;
    relearn::double_q_learning<state,action> learner(0.9, discount, seed + 50);
    learner.learning_rate = relearn::polynomial_decay<>(1.0, 1.0);
    for (int k = 1; k <= episodes; k++) {
        learner(sample(prng), memory);
    }
    return memory;
}

/// verifies that `double_q_learning` reaches the exact Q of both MDPs (Sutton & Barto, Section 6.7)
TEST_CASE("double_q_learning converges to the true Q on stochastic MDPs", "[convergence][double_q]")
{
    // each estimate gets about half of the updates, so each counts its own: α = 1/n for
    // the n-th update of a pair in that estimate. On MDP B, across seeds 1 ... 10, the worst
    // error was 0.125 with α = 1/k per episode, 0.0075 with one count shared by both
    // estimates, and 0.0023 with a count per estimate (0.0034 on MDP A)
    const int n = 200000;
    SECTION("MDP A: stochastic final reward") {
        auto memory = train_double_q(mdp_a{}, n, 6);
        REQUIRE(memory.value(state("s1"), action("b")) == Approx(mdp_a{}.q_s1).margin(0.02));
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(mdp_a{}.q_s0).margin(0.02));
    }
    SECTION("MDP B: stochastic transition, intermediate reward") {
        auto memory = train_double_q(mdp_b{}, n, 6);
        REQUIRE(memory.value(state("s1"), action("x")) == Approx(mdp_b{}.q_s1).margin(0.02));
        REQUIRE(memory.value(state("s2"), action("y")) == Approx(mdp_b{}.q_s2).margin(0.02));
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(mdp_b{}.q_s0).margin(0.02));
    }
}
