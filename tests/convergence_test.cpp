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
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;

static const double discount = 0.9;

/// discounted return of a whole episode, starting at its first link
double discounted_return(const episode & ep)
{
    double g = 0, k = 1;
    for (const auto & link : ep) {
        g += k * link.state.reward();
        k *= discount;
    }
    return g;
}

/**
 * MDP A - stochastic outcome of the final decision
 *
 *   s0 --a--> s1 (r = 0);  s1 --b--> end with r = +1 (p = 0.7) or r = -1 (p = 0.3)
 *
 *   Q(s1,b) = 0.7 - 0.3           = 0.4
 *   Q(s0,a) = 0 + γ * Q(s1,b)     = 0.36
 */
struct mdp_a
{
    double q_s0 = discount * 0.4;
    double q_s1 = 0.4;

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
 * MDP B - stochastic transition and an intermediate reward
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
    double q_s0 = 0.5 + discount * (0.6 * 0.8 + 0.4 * -1.0);
    double q_s1 = 0.8;
    double q_s2 = -1.0;

    episode operator()(std::mt19937 & prng) const
    {
        std::bernoulli_distribution to_s1(0.6), win(0.8);
        if (to_s1(prng)) {
            return {{state(0.5, "s0"), action("a")},
                    {state(win(prng) ? 1 : 0, "s1"), action("x")}};
        }
        return {{state(0.5, "s0"), action("a")},
                {state(-1, "s2"), action("y")}};
    }
};

TEST_CASE("ground truth: exact Q matches the Monte Carlo return", "[convergence]")
{
    std::mt19937 prng(1);
    const int n = 200000;
    double g_a = 0, g_b = 0;
    for (int i = 0; i < n; i++) {
        g_a += discounted_return(mdp_a{}(prng));
        g_b += discounted_return(mdp_b{}(prng));
    }
    REQUIRE(g_a / n == Approx(mdp_a{}.q_s0).margin(0.01));
    REQUIRE(g_b / n == Approx(mdp_b{}.q_s0).margin(0.01));
}

TEST_CASE("q_learning: deterministic chain gives exact discounted values", "[convergence]")
{
    // s0 -> s1 -> s2, reward 1 for the final decision only
    episode chain = {{state(0, "s0"), action("a")},
                     {state(0, "s1"), action("b")},
                     {state(1, "s2"), action("c")}};
    relearn::policy<state,action> memory;
    relearn::q_learning<state,action> learner;
    learner.alpha = 1;      // deterministic: no averaging needed
    learner.gamma = discount;
    for (int i = 0; i < 3; i++) {
        learner(chain, memory);
    }
    REQUIRE(memory.value(state("s2"), action("c")) == Approx(1.0));
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(discount));
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(discount * discount));
}

/// train a q_learning learner with a decaying learning rate α = 1/k
template <class mdp>
relearn::policy<state,action> train_q_learning(const mdp & sample, int episodes)
{
    std::mt19937 prng(2);
    relearn::policy<state,action> memory;
    relearn::q_learning<state,action> learner;
    learner.gamma = discount;
    for (int k = 1; k <= episodes; k++) {
        learner.alpha = 1.0 / k;
        learner(sample(prng), memory);
    }
    return memory;
}

/// train a q_probabilistic learner (model-based, no learning rate)
template <class mdp>
relearn::policy<state,action> train_q_probabilistic(const mdp & sample, int episodes)
{
    std::mt19937 prng(3);
    relearn::policy<state,action> memory;
    relearn::q_probabilistic<state,action> learner(discount);
    for (int k = 1; k <= episodes; k++) {
        learner(sample(prng), memory);
    }
    return memory;
}

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
