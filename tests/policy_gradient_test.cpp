// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/short_corridor.hpp"
#include "../examples/games/random_walk.hpp"
#include "../examples/games/access_control.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of the policy gradient methods (Sutton & Barto, Chapter 13):
 * - `softmax_policy`: π(a | s) from the preferences, ∇ ln π, and sampling
 * - `reinforce`, `reinforce_with_baseline` and `actor_critic`: updates worked out by hand,
 *   including the γ^t factor
 * - the short corridor (Example 13.1): REINFORCE (Figure 13.1), REINFORCE with a baseline
 *   (Figure 13.2), and Actor-Critic, against the exact value of the policy learnt
 * - `actor_critic_lambda` and `continuing_actor_critic`: updates worked out by hand, the book's
 *   pseudocode written out here step by step, λ = 0 against `actor_critic`, the critic against
 *   `td_lambda` and `differential_sarsa`, and the actor against `reinforce`; the short corridor
 *   where one-step Actor-Critic fails, and the access-control queue against its optimal policy
 * Random number generators are seeded, so results are reproducible.
 */

namespace
{
using state   = relearn::state<std::string>;
using action  = relearn::action<std::string>;
using step    = relearn::link<state,action>;
using episode = std::deque<step>;
using policy  = relearn::policy<state,action>;

/// @return the legal actions of the hand-worked tests: left and right, in every state
std::vector<action> left_right(const state &)
{
    return {action("left"), action("right")};
}

using corridor       = games::short_corridor;
using corridor_types = games::types<corridor>;
using corridor_reinforce = relearn::reinforce<corridor_types::state, corridor_types::action>;
using corridor_baseline  = relearn::reinforce_with_baseline<corridor_types::state, corridor_types::action>;
using corridor_critic    = relearn::actor_critic<corridor_types::state, corridor_types::action>;
using corridor_lambda    = relearn::actor_critic_lambda<corridor_types::state, corridor_types::action>;

/// @return the legal actions of the short corridor, for the learners: left and right, everywhere
std::vector<corridor_types::action> corridor_actions(const corridor_types::state &)
{
    return {corridor_types::action(corridor::go_left), corridor_types::action(corridor::go_right)};
}

/// what was learnt in the short corridor, over several runs
struct corridor_runs
{
    /// π(right) learnt in each run (in the first cell, when the cells are shown)
    std::vector<double> right;
    /// the exact value of the start, v(S), of the policy learnt in each run
    std::vector<double> values;
};

/**
 * @brief REINFORCE, with or without a baseline: play a whole episode, then learn from it
 * @return the played episode
 */
template <class learner_type,
          class chooser>
games::played<corridor> learn_episode(corridor & game,
                                      chooser && agent,
                                      learner_type & learner,
                                      corridor_types::policy & preferences)
{
    auto r = games::play(game, agent, 1000);
    learner(r.episode, preferences, r.terminated);
    return r;
}

/**
 * @brief Actor-Critic: play an episode, learning after every step
 * @return the played episode
 */
template <class chooser>
games::played<corridor> learn_episode(corridor & game,
                                      chooser && agent,
                                      corridor_critic & learner,
                                      corridor_types::policy & preferences)
{
    return games::play_online_actor_critic(game, agent, learner, preferences, 1000);
}

/**
 * @brief Actor-Critic with traces: play an episode, learning after every step
 * @return the played episode
 */
template <class chooser>
games::played<corridor> learn_episode(corridor & game,
                                      chooser && agent,
                                      corridor_lambda & learner,
                                      corridor_types::policy & preferences)
{
    return games::play_online_actor_critic(game, agent, learner, preferences, 1000);
}

/**
 * @brief learn the short corridor, from scratch, in several runs
 * @param fresh is the learner each run starts from (its step sizes, γ and legal actions)
 * @param shown is true for the agent to see which cell it is in
 * @param episodes is the number of episodes of each run (each one cut short after 1000 steps)
 * @param runs is the number of runs
 * @param seed seeds the runs: run i is seeded with 1000 * seed + i
 * @return the policy learnt in each run, and its exact value
 *
 * When the cells look the same, every run starts from π(right) = 0.05, the ε-greedy-left
 * policy (ε = 0.1) of Example 13.1, worth about -82: h(right) = -ln(19) / 2, h(left) = ln(19) / 2.
 * When they are shown, every run starts from the preferences 0, π(right) = 1/2 in every cell.
 */
template <class learner_type>
corridor_runs learn_corridor(const learner_type & fresh,
                             bool shown,
                             unsigned int episodes,
                             unsigned int runs,
                             unsigned int seed)
{
    corridor_runs result;
    const corridor_types::action right(corridor::go_right);
    for (unsigned int run = 0; run < runs; run++) {
        std::mt19937 prng(1000 * seed + run);
        corridor game;
        game.show_position(shown);
        learner_type learner = fresh;
        corridor_types::policy preferences;
        if (!shown) {
            const double half = std::log(19.0) / 2;
            preferences.update(corridor_types::state(0), right, -half);
            preferences.update(corridor_types::state(0), corridor_types::action(corridor::go_left), half);
        }
        // the agent draws its actions from π, the policy being learnt
        auto agent = [&](const corridor::state_type & s,
                         const std::vector<corridor::action_type> &) {
            return learner.sample(corridor_types::state(s), preferences, prng).trait();
        };
        for (unsigned int e = 0; e < episodes; e++) {
            learn_episode(game, agent, learner, preferences);
        }
        // π(right) in each cell, and the exact value of the policy learnt
        std::vector<double> p;
        for (unsigned int cell = 0; cell < 3; cell++) {
            p.push_back(learner.probability(corridor_types::state(shown ? cell : 0), right, preferences));
        }
        result.right.push_back(p[0]);
        result.values.push_back(shown ? corridor::start_value(p[0], p[1], p[2])
                                      : corridor::start_value(p[0]));
    }
    return result;
}

/// @return the median of @param values (the upper one of the two middle values, for an even count)
double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

/// @return how many of @param values are above @param threshold
std::size_t above(const std::vector<double> & values,
                  double threshold)
{
    return static_cast<std::size_t>(std::count_if(values.begin(), values.end(),
                                                  [threshold](double v) { return v > threshold; }));
}
} // namespace

/// verifies π(a | s), the softmax of the preferences
TEST_CASE("softmax_policy: the probabilities of the preferences", "[policy_gradient][softmax]")
{
    relearn::softmax_policy<state,action> pi(left_right);
    policy preferences;
    // a state never learnt: every preference is 0, so both actions are equally likely
    REQUIRE(pi.probability(state("s"), action("left"), preferences) == Approx(0.5));
    REQUIRE(pi.probability(state("s"), action("right"), preferences) == Approx(0.5));
    // h(right) = ln 3, h(left) = 0: π(right) = 3 / (3 + 1)
    preferences.update(state("s"), action("right"), std::log(3.0));
    REQUIRE(pi.probability(state("s"), action("right"), preferences) == Approx(0.75));
    REQUIRE(pi.probability(state("s"), action("left"), preferences) == Approx(0.25));
    auto all = pi.probabilities(state("s"), preferences);
    REQUIRE(all.size() == 2);
    REQUIRE(all[0].first == action("left"));
    REQUIRE(all[0].second + all[1].second == Approx(1));
    // an action that is not legal is never taken
    REQUIRE(pi.probability(state("s"), action("up"), preferences) == 0);
    // huge preferences do not overflow: e^1000 does not fit in a double, but only
    // their difference counts, π(right) = 1 / (1 + e^-1)
    preferences.update(state("t"), action("right"), 1000);
    preferences.update(state("t"), action("left"), 999);
    REQUIRE(pi.probability(state("t"), action("right"), preferences) == Approx(1 / (1 + std::exp(-1.0))));
    REQUIRE(pi.probability(state("t"), action("left"), preferences) == Approx(1 / (1 + std::exp(1.0))));
    // and the highest preference is the one taken away, whichever action it belongs to: with 0 and 1000,
    // taking away the first one (0) would leave e^1000, and π would be inf / inf, not a number
    preferences.update(state("u"), action("right"), 1000);
    preferences.update(state("u"), action("left"), 0);
    preferences.update(state("w"), action("right"), 0);
    preferences.update(state("w"), action("left"), 1000);
    for (const auto & s : {"u", "w"}) {
        const auto p_right = pi.probability(state(s), action("right"), preferences);
        const auto p_left  = pi.probability(state(s), action("left"), preferences);
        REQUIRE_FALSE(std::isnan(p_right));
        REQUIRE_FALSE(std::isnan(p_left));
        REQUIRE(p_right + p_left == Approx(1));
    }
    REQUIRE(pi.probability(state("u"), action("right"), preferences) == Approx(1));
    REQUIRE(pi.probability(state("w"), action("left"), preferences) == Approx(1));
}

/// verifies ∇ ln π, and the step along it
TEST_CASE("softmax_policy: the gradient of ln π", "[policy_gradient][softmax]")
{
    relearn::softmax_policy<state,action> pi(left_right);
    policy preferences;
    preferences.update(state("s"), action("right"), std::log(3.0));
    // ∂ ln π(left | s) / ∂ h(s, b) = 1[left = b] - π(b | s): 1 - 0.25 for left, 0 - 0.75 for right
    auto gradient = pi.log_gradient(state("s"), action("left"), preferences);
    REQUIRE(gradient.size() == 2);
    REQUIRE(gradient[0].first == action("left"));
    REQUIRE(gradient[0].second == Approx(0.75));
    REQUIRE(gradient[1].second == Approx(-0.75));
    // a step of 2 along it: h(left) = 0 + 2 * 0.75, h(right) = ln 3 - 2 * 0.75
    pi.ascend(state("s"), action("left"), 2, preferences);
    REQUIRE(preferences.value(state("s"), action("left")) == Approx(1.5));
    REQUIRE(preferences.value(state("s"), action("right")) == Approx(std::log(3.0) - 1.5));
    // the preferences of other states do not move
    REQUIRE(preferences.actions(state("t")).empty());
}

/// verifies that the actions drawn follow π
TEST_CASE("softmax_policy: sample draws from π", "[policy_gradient][softmax]")
{
    relearn::softmax_policy<state,action> pi(left_right);
    policy preferences;
    preferences.update(state("s"), action("right"), std::log(3.0));
    std::mt19937 prng(1);
    // π(right) = 0.75: about 75000 of 100000 draws go right (the standard deviation is 137)
    int rights = 0;
    for (int i = 0; i < 100000; i++) {
        rights += pi.sample(state("s"), preferences, prng) == action("right");
    }
    REQUIRE(rights > 74000);
    REQUIRE(rights < 76000);
    // a huge preference: always that action
    preferences.update(state("t"), action("left"), 1000);
    for (int i = 0; i < 1000; i++) {
        REQUIRE(pi.sample(state("t"), preferences, prng) == action("left"));
    }
}

/// verifies one REINFORCE episode worked out by hand, with the γ^t factor
TEST_CASE("reinforce: one episode by hand", "[policy_gradient][reinforce]")
{
    // α = 0.5, γ = 0.5: (s0, right) earns 0, then (s1, left) earns 2.
    // The returns are G_1 = 2 and G_0 = 0 + 0.5 * 2 = 1, and π is 1/2 everywhere at first.
    //    t = 0: α γ^0 G_0 = 0.5, so h(s0, right) = 0.5 * (1 - 1/2) = 0.25, h(s0, left) = -0.25
    //    t = 1: α γ^1 G_1 = 0.5, so h(s1, left)  = 0.5 * (1 - 1/2) = 0.25, h(s1, right) = -0.25
    // (without γ^t, the step at t = 1 would be α G_1 = 1, and h(s1, left) 0.5)
    relearn::reinforce<state,action> learner(0.5, 0.5, left_right);
    policy preferences;
    learner(episode {{state(0, "s0"), action("right")},
                     {state(2, "s1"), action("left")}}, preferences);
    REQUIRE(preferences.value(state("s0"), action("right")) == Approx(0.25));
    REQUIRE(preferences.value(state("s0"), action("left")) == Approx(-0.25));
    REQUIRE(preferences.value(state("s1"), action("left")) == Approx(0.25));
    REQUIRE(preferences.value(state("s1"), action("right")) == Approx(-0.25));
    // the policy learnt: right is now more likely in s0, π(right | s0) = 1 / (1 + e^-0.5)
    REQUIRE(learner.probability(state("s0"), action("right"), preferences) == Approx(0.6224593));
}

/// verifies that θ changes from one step to the next whilst the episode is walked
TEST_CASE("reinforce: each step starts from the preferences left by the one before", "[policy_gradient][reinforce]")
{
    // α = 0.5, γ = 1: (s, right) twice, each earning -1. The returns are G_0 = -2 and G_1 = -1.
    //    t = 0: π(right) = 1/2, α G_0 = -1: h(right) = -1 * (1 - 1/2) = -0.5, h(left) = 0.5
    //    t = 1: π(right) = e^-0.5 / (e^-0.5 + e^0.5) = 1 / (1 + e) = 0.2689414, α G_1 = -0.5:
    //           h(right) = -0.5 - 0.5 * (1 - 0.2689414) = -0.8655293, h(left) = 0.8655293
    // (with π still at 1/2 for t = 1, h(right) would be -0.75)
    relearn::reinforce<state,action> learner(0.5, 1.0, left_right);
    policy preferences;
    learner(episode {{state(-1, "s"), action("right")},
                     {state(-1, "s"), action("right")}}, preferences);
    REQUIRE(preferences.value(state("s"), action("right")) == Approx(-0.8655293));
    REQUIRE(preferences.value(state("s"), action("left")) == Approx(0.8655293));
}

/// verifies that an episode cut short is not learnt
TEST_CASE("reinforce: an episode cut short is not learnt", "[policy_gradient][reinforce]")
{
    relearn::reinforce<state,action> plain(0.5, 1.0, left_right);
    relearn::reinforce_with_baseline<state,action> baseline(0.5, 0.5, 1.0, left_right);
    policy preferences;
    auto cut = episode {{state(-1, "s0"), action("right")},
                        {state(0, "s1"), action("left")}};
    plain(cut, preferences, false);
    baseline(cut, preferences, false);
    REQUIRE(preferences.actions(state("s0")).empty());
    REQUIRE(baseline.value(state("s0")) == 0);
}

/// verifies two episodes of REINFORCE with a baseline, worked out by hand
TEST_CASE("reinforce_with_baseline: two episodes by hand", "[policy_gradient][baseline]")
{
    // α^θ = 0.5, α^w = 0.5, γ = 0.5: (s0, right) earns 0, then (s1, left) earns 2, twice.
    // The returns are G_0 = 1 and G_1 = 2.
    relearn::reinforce_with_baseline<state,action> learner(0.5, 0.5, 0.5, left_right);
    policy preferences;
    auto twice = episode {{state(0, "s0"), action("right")},
                          {state(2, "s1"), action("left")}};
    // 1. t = 0: δ = 1 - 0 = 1, v(s0) = 0.5; α^θ γ^0 δ = 0.5, so h(s0, right) = 0.5 * (1 - 1/2) = 0.25
    //    t = 1: δ = 2 - 0 = 2, v(s1) = 1;   α^θ γ^1 δ = 0.5, so h(s1, left)  = 0.25
    learner(twice, preferences);
    REQUIRE(learner.value(state("s0")) == Approx(0.5));
    REQUIRE(learner.value(state("s1")) == Approx(1));
    REQUIRE(preferences.value(state("s0"), action("right")) == Approx(0.25));
    REQUIRE(preferences.value(state("s0"), action("left")) == Approx(-0.25));
    REQUIRE(preferences.value(state("s1"), action("left")) == Approx(0.25));
    // 2. now π(right | s0) = π(left | s1) = 1 / (1 + e^-0.5) = 0.6224593
    //    t = 0: δ = 1 - 0.5 = 0.5, v(s0) = 0.75; α^θ γ^0 δ = 0.25,
    //           so h(s0, right) = 0.25 + 0.25 * (1 - 0.6224593) = 0.3443852
    //    t = 1: δ = 2 - 1 = 1, v(s1) = 1.5;      α^θ γ^1 δ = 0.25, so h(s1, left) = 0.3443852
    learner(twice, preferences);
    REQUIRE(learner.value(state("s0")) == Approx(0.75));
    REQUIRE(learner.value(state("s1")) == Approx(1.5));
    REQUIRE(preferences.value(state("s0"), action("right")) == Approx(0.3443852));
    REQUIRE(preferences.value(state("s0"), action("left")) == Approx(-0.3443852));
    REQUIRE(preferences.value(state("s1"), action("left")) == Approx(0.3443852));
    REQUIRE(preferences.value(state("s1"), action("right")) == Approx(-0.3443852));
}

/// verifies two episodes of one-step Actor-Critic, worked out by hand, with I = γ^t
TEST_CASE("actor_critic: two episodes by hand", "[policy_gradient][actor_critic]")
{
    // α^θ = 0.5, α^w = 0.5, γ = 0.5: (s0, right) earns 1 and leads to s1, then (s1, left) earns 2 and ends.
    relearn::actor_critic<state,action> learner(0.5, 0.5, 0.5, left_right);
    policy preferences;
    auto twice = episode {{state(1, "s0"), action("right")},
                          {state(2, "s1"), action("left")}};
    // 1. t = 0: I = 1,   δ = 1 + 0.5 * v(s1) - v(s0) = 1, v(s0) = 0.5; α^θ I δ = 0.5, so h(s0, right) = 0.25
    //    t = 1: I = 0.5, δ = 2 - v(s1) = 2 (the end),      v(s1) = 1;   α^θ I δ = 0.5, so h(s1, left)  = 0.25
    REQUIRE(learner.discount() == 1);
    learner(twice, preferences);
    REQUIRE(learner.value(state("s0")) == Approx(0.5));
    REQUIRE(learner.value(state("s1")) == Approx(1));
    REQUIRE(preferences.value(state("s0"), action("right")) == Approx(0.25));
    REQUIRE(preferences.value(state("s0"), action("left")) == Approx(-0.25));
    REQUIRE(preferences.value(state("s1"), action("left")) == Approx(0.25));
    REQUIRE(preferences.value(state("s1"), action("right")) == Approx(-0.25));
    // the episode ended, so I is 1 again
    REQUIRE(learner.discount() == 1);
    // 2. now π(right | s0) = π(left | s1) = 0.6224593
    //    t = 0: δ = 1 + 0.5 * 1 - 0.5 = 1, v(s0) = 1;  α^θ I δ = 0.5,  so h(s0, right) = 0.25 + 0.5 * 0.3775407 = 0.4387704
    //    t = 1: δ = 2 - 1 = 1,             v(s1) = 1.5; α^θ I δ = 0.25, so h(s1, left)  = 0.25 + 0.25 * 0.3775407 = 0.3443852
    learner(twice, preferences);
    REQUIRE(learner.value(state("s0")) == Approx(1));
    REQUIRE(learner.value(state("s1")) == Approx(1.5));
    REQUIRE(preferences.value(state("s0"), action("right")) == Approx(0.4387704));
    REQUIRE(preferences.value(state("s1"), action("left")) == Approx(0.3443852));
}

/// verifies that learning step by step, as the episode is played, learns the same as the whole episode
TEST_CASE("actor_critic: step by step, and episodes cut short", "[policy_gradient][actor_critic]")
{
    // the same episode as above, learnt all at once by one learner, and step by step by another
    relearn::actor_critic<state,action> whole(0.5, 0.5, 0.5, left_right);
    relearn::actor_critic<state,action> stepwise(0.5, 0.5, 0.5, left_right);
    policy whole_preferences;
    policy stepwise_preferences;
    whole(episode {{state(1, "s0"), action("right")},
                   {state(2, "s1"), action("left")}}, whole_preferences);
    // step 0, learnt from the state reached (the action of the second link is not used)
    stepwise(episode {{state(1, "s0"), action("right")},
                      {state(0, "s1"), action("right")}}, stepwise_preferences, false);
    REQUIRE(stepwise.discount() == Approx(0.5));
    REQUIRE(stepwise.value(state("s1")) == 0);
    // step 1, the end of the episode
    stepwise(episode {{state(2, "s1"), action("left")}}, stepwise_preferences, true);
    REQUIRE(stepwise.discount() == 1);
    for (const auto & s : {"s0", "s1"}) {
        REQUIRE(stepwise.value(state(s)) == Approx(whole.value(state(s))));
        for (const auto & a : {"left", "right"}) {
            REQUIRE(stepwise_preferences.value(state(s), action(a)) ==
                    Approx(whole_preferences.value(state(s), action(a))));
        }
    }
    // the same episode again, now that v(s1) = 1: step 0 bootstraps from it,
    // δ = 1 + 0.5 * v(s1) - v(s0) = 1 + 0.5 - 0.5 = 1, as in the second episode of the test above
    whole(episode {{state(1, "s0"), action("right")},
                   {state(2, "s1"), action("left")}}, whole_preferences);
    stepwise(episode {{state(1, "s0"), action("right")},
                      {state(0, "s1"), action("right")}}, stepwise_preferences, false);
    stepwise(episode {{state(2, "s1"), action("left")}}, stepwise_preferences, true);
    REQUIRE(stepwise.value(state("s0")) == Approx(1));
    for (const auto & s : {"s0", "s1"}) {
        REQUIRE(stepwise.value(state(s)) == Approx(whole.value(state(s))));
        for (const auto & a : {"left", "right"}) {
            REQUIRE(stepwise_preferences.value(state(s), action(a)) ==
                    Approx(whole_preferences.value(state(s), action(a))));
        }
    }
    // an episode cut short leaves I where it was, until the next episode starts
    stepwise(episode {{state(1, "s0"), action("right")},
                      {state(0, "s1"), action("left")}}, stepwise_preferences, false);
    REQUIRE(stepwise.discount() == Approx(0.5));
    stepwise.start_episode();
    REQUIRE(stepwise.discount() == 1);
}

/// verifies the constructors, and brace initialisation
TEST_CASE("policy gradient: constructors and brace initialisation", "[policy_gradient]")
{
    relearn::reinforce<state,action> plain;
    REQUIRE(plain.alpha == Approx(0.01));
    REQUIRE(plain.gamma == Approx(0.9));
    REQUIRE_FALSE(plain.legal_actions);
    auto baseline = relearn::reinforce_with_baseline<state,action>{0.25, 0.5, 1.0, left_right};
    REQUIRE(baseline.alpha_theta == Approx(0.25));
    REQUIRE(baseline.alpha_w == Approx(0.5));
    REQUIRE(baseline.gamma == Approx(1));
    REQUIRE(baseline.legal_actions(state("s")).size() == 2);
    auto critic = relearn::actor_critic<state,action>{0.25, 0.5, 1.0, left_right};
    REQUIRE(critic.alpha_theta == Approx(0.25));
    REQUIRE(critic.alpha_w == Approx(0.5));
    REQUIRE(critic.discount() == 1);
    REQUIRE(critic.value(state("s")) == 0);
}

/// verifies Figure 13.1: REINFORCE learns close to the best stochastic policy of the short corridor
TEST_CASE("reinforce: the short corridor (Figure 13.1)", "[policy_gradient][reinforce][short_corridor]")
{
    // α = 2^-13 (the best step size of Figure 13.1), γ = 1, 50 runs of 1000 episodes, from
    // π(right) = 0.05, worth -82. The best policy is p = 0.5858, worth -11.657, and REINFORCE
    // comes to it from below. Over 10 seeds (the same numbers with libstdc++ and with libc++),
    // the median p learnt was 0.470 to 0.486, the median value -12.29 to -12.12, and every run but one
    // was worth more than -13.8. That one run (seed 10) had an unlucky 17th episode, 651 steps
    // mostly bumping left into the wall, whose large returns pushed p from 0.055 to 1 at once;
    // from there every episode is cut short at 1000 steps, and so never learnt
    corridor_reinforce learner(std::pow(2.0, -13), 1.0, corridor_actions);
    auto learnt = learn_corridor(learner, false, 1000, 50, 1);
    REQUIRE(median(learnt.right) > 0.42);
    REQUIRE(median(learnt.right) < 0.65);
    REQUIRE(median(learnt.values) > -12.7);
    REQUIRE(above(learnt.values, -14) >= 48);
}

/// verifies Figure 13.2: with a baseline, REINFORCE learns much faster
TEST_CASE("reinforce_with_baseline: the short corridor (Figure 13.2)", "[policy_gradient][baseline][short_corridor]")
{
    // α^θ = 2^-9 and α^w = 2^-6 with a baseline, against α = 2^-13 without (Figure 13.2),
    // γ = 1, the same 50 runs, from π(right) = 0.05. Over 10 seeds (the same numbers with
    // libstdc++ and with libc++), after 200 episodes the median value of the policies learnt was
    // -11.92 to -11.73 with the baseline, and -21.59 to -20.65 without. After 1000 episodes
    // with the baseline, the median p was 0.577 to 0.617, the median value -11.84 to -11.73, and
    // at least 48 runs of 50 were worth more than -15 (the step size is large, so a few runs
    // wander off for a while)
    corridor_baseline baseline(std::pow(2.0, -9), std::pow(2.0, -6), 1.0, corridor_actions);
    corridor_reinforce plain(std::pow(2.0, -13), 1.0, corridor_actions);
    auto with = learn_corridor(baseline, false, 200, 50, 1);
    auto without = learn_corridor(plain, false, 200, 50, 1);
    REQUIRE(median(with.values) > -12.3);
    REQUIRE(median(without.values) < -19);
    // and it stays close to the best policy
    auto longer = learn_corridor(baseline, false, 1000, 50, 1);
    REQUIRE(median(longer.right) > 0.53);
    REQUIRE(median(longer.right) < 0.65);
    REQUIRE(median(longer.values) > -12.3);
    REQUIRE(above(longer.values, -15) >= 46);
}

/// verifies that Actor-Critic learns the best policy when it can tell the cells apart
TEST_CASE("actor_critic: the short corridor, with the cells shown", "[policy_gradient][actor_critic][short_corridor]")
{
    // the agent sees its cell, so the best policy is right, left, right, worth -3.
    // α^θ = 2^-4, α^w = 2^-3, γ = 1, 20 runs of 1000 episodes, from π(right) = 1/2 everywhere.
    // Over 10 seeds (the same numbers with libstdc++ and with libc++), every run learnt a policy
    // worth -3.039 to -3.014, with π(right) = 0.986 to 0.997 in the first cell
    corridor_critic learner(std::pow(2.0, -4), std::pow(2.0, -3), 1.0, corridor_actions);
    auto learnt = learn_corridor(learner, true, 1000, 20, 1);
    REQUIRE(above(learnt.values, -3.1) == learnt.values.size());
    REQUIRE(median(learnt.right) > 0.97);
}

/// verifies that Actor-Critic is misled by a critic which cannot tell the cells apart
TEST_CASE("actor_critic: the short corridor, with the cells looking the same", "[policy_gradient][actor_critic][short_corridor]")
{
    // The critic has one value v for all three cells, so the TD error of every step that does not
    // reach G is δ = -1 + v - v = -1, whichever action was taken: on average it does not move the
    // policy. Only the step onto G, right in the third cell, has δ = -1 - v, more than the -1 of
    // left there as soon as v < 0, and so right becomes more likely. p keeps growing, past the best
    // p = 0.5858, towards 1, where the agent goes back and forth around the second cell.
    // The step sizes of Figure 13.2, α^θ = 2^-9 and α^w = 2^-6, γ = 1, 5 runs of 1000 episodes,
    // from π(right) = 0.05. Over 10 seeds (the same numbers with libstdc++ and with libc++),
    // every run learnt p = 0.9981 to 0.9986, worth -1449 to -1074
    corridor_critic learner(std::pow(2.0, -9), std::pow(2.0, -6), 1.0, corridor_actions);
    auto learnt = learn_corridor(learner, false, 1000, 5, 1);
    for (std::size_t run = 0; run < learnt.right.size(); run++) {
        REQUIRE(learnt.right[run] > 0.99);
        REQUIRE(learnt.values[run] < -500);
    }
}

namespace
{
/// the book's parameters of an Actor-Critic with traces, for the reference below
struct book_parameters
{
    double alpha_theta;
    double alpha_w;
    double gamma;
    double lambda_theta;
    double lambda_w;
    /// the step size of the average reward, for the continuing one
    double beta;
    /// true for the continuing one (Section 13.6), false for the episodic one (Section 13.5)
    bool continuing;
};

/**
 * @struct book_actor_critic
 * @brief Sutton & Barto's pseudocode of Actor-Critic with eligibility traces (Sections 13.5 and 13.6), written
 *        out here plainly with maps, one feature per state for v̂ and per state/action for the preferences,
 *        and its own softmax: the reference the learners are checked against, step by step
 */
struct book_actor_critic
{
    book_parameters p;
    std::map<std::string,double> w;
    std::map<std::pair<std::string,std::string>,double> theta;
    std::map<std::string,double> z_w;
    std::map<std::pair<std::string,std::string>,double> z_theta;
    double I = 1;
    double R_bar = 0;

    /// @brief a new episode: z^w = 0, z^θ = 0, I = 1
    void start()
    {
        z_w.clear();
        z_theta.clear();
        I = 1;
    }

    /// @return π(a | s) for the legal actions a of s, the softmax of their preferences: left and right, but only left in s3
    std::map<std::string,double> pi(const std::string & s) const
    {
        auto h = [&](const std::string & a) {
            auto found = theta.find({s, a});
            return found == theta.end() ? 0.0 : found->second;
        };
        if (s == "s3") {
            return {{"left", 1.0}};
        }
        const double left = std::exp(h("left"));
        const double right = std::exp(h("right"));
        return {{"left", left / (left + right)}, {"right", right / (left + right)}};
    }

    /// @brief one step: in s, take a, receive r and reach s_next (none if the episode ended)
    void step(const std::string & s,
              const std::string & a,
              double r,
              const std::string * s_next)
    {
        const double v_next = s_next ? w[*s_next] : 0.0;
        const double discount = p.continuing ? 1.0 : p.gamma;
        double delta = r + discount * v_next - w[s];
        if (p.continuing) {
            delta = r - R_bar + v_next - w[s];
            R_bar += p.beta * delta;
        }
        // z^w = γλ^w z^w + ∇v̂(S, w),   z^θ = γλ^θ z^θ + I ∇ ln π(A | S, θ)   (no γ and no I when continuing)
        for (auto & z : z_w) {
            z.second *= discount * p.lambda_w;
        }
        z_w[s] += 1;
        const auto probabilities = pi(s);
        for (auto & z : z_theta) {
            z.second *= discount * p.lambda_theta;
        }
        for (const auto & b : probabilities) {
            z_theta[{s, b.first}] += (p.continuing ? 1.0 : I) * ((b.first == a ? 1.0 : 0.0) - b.second);
        }
        // w = w + α^w δ z^w,   θ = θ + α^θ δ z^θ
        for (const auto & z : z_w) {
            w[z.first] += p.alpha_w * delta * z.second;
        }
        for (const auto & z : z_theta) {
            theta[z.first] += p.alpha_theta * delta * z.second;
        }
        I *= p.gamma;
    }
};

/// @return the legal actions of the pseudocode test: left and right, but only left in s3
std::vector<action> mostly_left_right(const state & s)
{
    return s.trait() == "s3" ? std::vector<action> {action("left")} : left_right(s);
}

/**
 * @return a random episode of 1 to 12 steps over the states s0 ... s3, which come back often, and the actions
 *         left and right (only left in s3, where it is the only legal action of `mostly_left_right`)
 */
episode random_episode(std::mt19937 & prng)
{
    std::uniform_int_distribution<int> pick_state(0, 3);
    std::uniform_int_distribution<int> pick_action(0, 1);
    std::uniform_int_distribution<int> pick_reward(-2, 3);
    std::uniform_int_distribution<int> pick_length(1, 12);
    episode e;
    const int length = pick_length(prng);
    for (int i = 0; i < length; i++) {
        const std::string s = "s" + std::to_string(pick_state(prng));
        const bool left = pick_action(prng) == 0 || s == "s3";
        e.push_back({state(pick_reward(prng), s), action(left ? "left" : "right")});
    }
    return e;
}

/**
 * @brief the largest difference between a learner's values and preferences and the reference's
 * @param learner is `actor_critic_lambda` or `continuing_actor_critic`
 */
template <class learner_type>
double largest_difference(const learner_type & learner,
                          const policy & preferences,
                          const book_actor_critic & book)
{
    double largest = 0;
    for (int k = 0; k < 4; k++) {
        const std::string s = "s" + std::to_string(k);
        auto found = book.w.find(s);
        largest = std::max(largest, std::fabs(learner.value(state(s)) - (found == book.w.end() ? 0.0 : found->second)));
        for (const auto & a : {"left", "right"}) {
            auto h = book.theta.find({s, a});
            largest = std::max(largest, std::fabs(preferences.value(state(s), action(a)) - (h == book.theta.end() ? 0.0 : h->second)));
        }
    }
    return largest;
}

using walk_type  = games::random_walk;
using walk_types = games::types<walk_type>;
using queue_type  = games::access_control;
using queue_types = games::types<queue_type>;

/// @return the legal actions of the queue, for the learners: reject, and accept if a server is free
std::vector<queue_types::action> queue_actions(const queue_types::state & s)
{
    queue_type rules;
    std::vector<queue_types::action> legal;
    for (auto a : rules.actions(s.trait())) {
        legal.push_back(queue_types::action(a));
    }
    return legal;
}
} // namespace

/// verifies Actor-Critic with traces, episodic, worked out by hand
TEST_CASE("actor_critic_lambda: one episode by hand", "[policy_gradient][actor_critic_lambda]")
{
    // α^θ = α^w = 0.5, γ = 0.5, λ^θ = λ^w = 0.5 (so γλ = 0.25): (x, right) earns 1, (y, left) 0, then (x, right) 2 and ends.
    //    t = 0: I = 1,    δ = 1,                          z^w(x) = 1,  v(x) = 0.5
    //           z^θ(x, right) = 1 * (1 - 0.5) = 0.5, z^θ(x, left) = -0.5, so h(x, right) = 0.25, h(x, left) = -0.25
    //    t = 1: I = 0.5,  δ = 0 + 0.5 v(x) - v(y) = 0.25, z^w(x) = 0.25, z^w(y) = 1: v(x) = 0.53125, v(y) = 0.125
    //           z^θ(x, ·) = ±0.125, z^θ(y, left) = 0.5 * 0.5 = 0.25, z^θ(y, right) = -0.25, α^θ δ = 0.125:
    //           h(x, right) = 0.265625, h(y, left) = 0.03125
    //    t = 2: I = 0.25, δ = 2 - v(x) = 1.46875,         z^w(x) = 1.0625, z^w(y) = 0.25: v(x) = 1.3115234375, v(y) = 0.30859375
    //           (the very values of SARSA(λ)'s test with accumulating traces: the critic is TD(λ)),
    //           π(right | x) = 1 / (1 + e^-0.53125), z^θ(x, right) = 0.03125 + 0.25 (1 - π(right | x)),
    //           z^θ(y, left) = 0.0625; h(x, right) = 0.265625 + 0.734375 z^θ(x, right), h(y, left) = 0.03125 + 0.734375 * 0.0625
    // (the expected numbers were worked out apart from relearn, in a few lines of Python)
    relearn::actor_critic_lambda<state,action> learner(0.5, 0.5, 0.5, 0.5, 0.5, left_right);
    policy preferences;
    learner(episode {{state(1, "x"), action("right")},
                     {state(0, "y"), action("left")},
                     {state(2, "x"), action("right")}}, preferences);
    REQUIRE(learner.value(state("x")) == Approx(1.3115234375));
    REQUIRE(learner.value(state("y")) == Approx(0.30859375));
    REQUIRE(learner.critic_trace(state("x")) == Approx(1.0625));
    REQUIRE(learner.critic_trace(state("y")) == Approx(0.25));
    REQUIRE(learner.actor_traces.trace(state("x"), action("right")) == Approx(0.12380634814889666));
    REQUIRE(learner.actor_traces.trace(state("y"), action("left")) == Approx(0.0625));
    REQUIRE(preferences.value(state("x"), action("right")) == Approx(0.35654528692184595));
    REQUIRE(preferences.value(state("x"), action("left")) == Approx(-0.35654528692184595));
    REQUIRE(preferences.value(state("y"), action("left")) == Approx(0.0771484375));
    REQUIRE(preferences.value(state("y"), action("right")) == Approx(-0.0771484375));
    // the episode ended: I is 1 again
    REQUIRE(learner.discount() == 1);
}

/// verifies Actor-Critic with traces for a task that never ends, worked out by hand
TEST_CASE("continuing_actor_critic: three steps by hand", "[policy_gradient][continuing_actor_critic]")
{
    // α^θ = α^w = β = 0.5, λ^θ = λ^w = 0.5, R̄ = 0: (x, right) earns 2, (y, left) 0, (x, left) 1, then y is reached.
    //    t = 0: δ = 2 - 0 + 0 - 0 = 2, R̄ = 1; z^w(x) = 1, v(x) = 1; z^θ(x, right) = 0.5, so h(x, right) = 0.5
    //    t = 1: δ = 0 - 1 + v(x) - v(y) = 0, R̄ = 1: nothing moves, the traces fade by λ = 0.5 and z^w(y) = 1
    //    t = 2: π(left | x) = 1 / (1 + e) = 0.2689, δ = 1 - 1 + v(y) - v(x) = -1, R̄ = 0.5;
    //           z^w(x) = 1.25, z^w(y) = 0.5, so v(x) = 0.375, v(y) = -0.25; z^θ(x, left) = -0.125 + 0.7311 = 0.6061
    // (worked out apart from relearn, in a few lines of Python)
    relearn::continuing_actor_critic<state,action> learner(0.5, 0.5, 0.5, 0.5, 0.5, left_right);
    policy preferences;
    learner(episode {{state(2, "x"), action("right")},
                     {state(0, "y"), action("left")},
                     {state(1, "x"), action("left")},
                     {state(0, "y"), action("right")}}, preferences);
    REQUIRE(learner.rho == Approx(0.5));
    REQUIRE(learner.value(state("x")) == Approx(0.375));
    REQUIRE(learner.value(state("y")) == Approx(-0.25));
    REQUIRE(learner.critic_trace(state("x")) == Approx(1.25));
    REQUIRE(learner.critic_trace(state("y")) == Approx(0.5));
    REQUIRE(learner.actor_traces.trace(state("x"), action("left")) == Approx(0.6060585786300049));
    REQUIRE(preferences.value(state("x"), action("right")) == Approx(0.8030292893150024));
    REQUIRE(preferences.value(state("y"), action("right")) == Approx(0.125));
    // the last link, y, is only the state reached: it has no trace of its own yet, and it goes on from here
    learner(episode {{state(0, "y"), action("right")},
                     {state(0, "x"), action("left")}}, preferences);
    REQUIRE(learner.critic_trace(state("y")) == Approx(1.25));
}

/// verifies both learners against the book's pseudocode, written out here, after every step
TEST_CASE("actor_critic_lambda and continuing_actor_critic: the book's pseudocode, step by step", "[policy_gradient][actor_critic_lambda][continuing_actor_critic]")
{
    // 300 pairs of random episodes that come back to the same states, with random step sizes, γ and λs (γ = 1 and
    // λ = 1 too), each pair learnt one after the other by the same learner, some ending and some cut short, and a state,
    // s3, with a single legal action. Learnt one step at a time (as `games::play_online_actor_critic` gives them) and
    // in one call per episode: after every step, the values, preferences and R̄ are those of the pseudocode.
    // For the task that never ends, an episode cut short goes on in the next one (which starts with its last link),
    // and one that ends starts its traces again, keeping R̄ and the values
    std::mt19937 prng(7);
    std::uniform_real_distribution<double> uniform(0, 1);
    double largest = 0;
    for (int trial = 0; trial < 300; trial++) {
        book_parameters p {0.05 + 0.5 * uniform(prng), 0.05 + 0.5 * uniform(prng),
                           trial % 5 == 0 ? 1.0 : uniform(prng),
                           trial % 7 == 0 ? 1.0 : uniform(prng), trial % 11 == 0 ? 1.0 : uniform(prng),
                           0.05 + 0.5 * uniform(prng), trial % 2 == 1};
        const std::vector<bool> ends = {trial % 3 != 0, trial % 4 != 0};
        std::vector<episode> episodes = {random_episode(prng), random_episode(prng)};
        if (p.continuing && !ends[0]) {
            // the task goes on: the next piece starts with the state reached and the action taken next
            episodes[1].front() = {state(episodes[1].front().state.reward(), episodes[0].back().state.trait()),
                                   episodes[0].back().action};
        }
        book_actor_critic book {p, {}, {}, {}, {}};
        relearn::actor_critic_lambda<state,action> episodic(p.alpha_theta, p.alpha_w, p.gamma, p.lambda_theta, p.lambda_w, mostly_left_right);
        relearn::continuing_actor_critic<state,action> continuing(p.alpha_theta, p.alpha_w, p.beta, p.lambda_theta, p.lambda_w, mostly_left_right);
        policy stepwise;
        for (std::size_t k = 0; k < 2; k++) {
            const auto & e = episodes[k];
            // a new episode for the reference: always when episodic, and after an episode that ended when continuing
            if (!p.continuing || k == 0 || ends[0]) {
                book.start();
            }
            if (!p.continuing) {
                episodic.start_episode();
            }
            const std::size_t T = ends[k] ? e.size() : e.size() - 1;
            for (std::size_t t = 0; t < T; t++) {
                const bool last = t + 1 == e.size();
                const std::string next = last ? "" : e[t + 1].state.trait();
                book.step(e[t].state.trait(), e[t].action.trait(), e[t].state.reward(), last ? nullptr : &next);
                episode window = last ? episode {e[t]} : episode {e[t], {state(0, next), e[t].action}};
                if (p.continuing) {
                    continuing(window, stepwise, last);
                    largest = std::max(largest, largest_difference(continuing, stepwise, book));
                    largest = std::max(largest, std::fabs(continuing.rho - book.R_bar));
                }
                else {
                    episodic(window, stepwise, last);
                    largest = std::max(largest, largest_difference(episodic, stepwise, book));
                }
            }
            if (p.continuing && ends[k]) {
                // a task that ended: its traces start again
                REQUIRE(continuing.actor_traces.entries().empty());
                for (int s = 0; s < 4; s++) {
                    REQUIRE(continuing.critic_trace(state("s" + std::to_string(s))) == 0);
                }
            }
        }
        // the single legal action of s3 has ∇ ln π = 1 - 1 = 0: its preference never moves
        REQUIRE(stepwise.value(state("s3"), action("left")) == 0);
        // each episode in one call: the very same, to the last bit
        policy at_once;
        relearn::continuing_actor_critic<state,action> whole_continuing(p.alpha_theta, p.alpha_w, p.beta, p.lambda_theta, p.lambda_w, mostly_left_right);
        relearn::actor_critic_lambda<state,action> whole_episodic(p.alpha_theta, p.alpha_w, p.gamma, p.lambda_theta, p.lambda_w, mostly_left_right);
        for (std::size_t k = 0; k < 2; k++) {
            if (p.continuing) {
                whole_continuing(episodes[k], at_once, ends[k]);
            }
            else {
                whole_episodic(episodes[k], at_once, ends[k]);
            }
        }
        if (p.continuing) {
            REQUIRE(whole_continuing.rho == continuing.rho);
        }
        for (int s = 0; s < 4; s++) {
            const state s_k("s" + std::to_string(s));
            REQUIRE((p.continuing ? whole_continuing.value(s_k) : whole_episodic.value(s_k)) ==
                    (p.continuing ? continuing.value(s_k) : episodic.value(s_k)));
            for (const auto & a : {"left", "right"}) {
                REQUIRE(at_once.value(s_k, action(a)) == stepwise.value(s_k, action(a)));
            }
        }
    }
    INFO("the largest difference is " << largest);
    REQUIRE(largest < 1e-12);
}

/// verifies that with λ^θ = λ^w = 0, Actor-Critic with traces is one-step Actor-Critic
TEST_CASE("actor_critic_lambda: lambda = 0 is actor_critic", "[policy_gradient][actor_critic_lambda]")
{
    // the same random episodes, some cut short, learnt by both (to within 1e-12: α^θ (δ I ∇) is rounded differently)
    std::mt19937 prng(8);
    relearn::actor_critic<state,action> one_step(0.3, 0.4, 0.9, left_right);
    relearn::actor_critic_lambda<state,action> zero(0.3, 0.4, 0.9, 0.0, 0.0, left_right);
    policy one_preferences, zero_preferences;
    for (int i = 0; i < 300; i++) {
        auto e = random_episode(prng);
        const bool terminated = i % 3 != 0;
        one_step.start_episode();
        one_step(e, one_preferences, terminated);
        zero(e, zero_preferences, terminated);
    }
    for (int k = 0; k < 4; k++) {
        const state s("s" + std::to_string(k));
        REQUIRE(zero.value(s) == Approx(one_step.value(s)).epsilon(1e-12));
        for (const auto & a : {"left", "right"}) {
            REQUIRE(zero_preferences.value(s, action(a)) == Approx(one_preferences.value(s, action(a))).epsilon(1e-12).margin(1e-15));
        }
    }
}

/// verifies the critic: with an actor that learns nothing, it is TD(λ), and in a task that never ends, differential TD(0)
TEST_CASE("actor_critic_lambda and continuing_actor_critic: the critic is TD(lambda)", "[policy_gradient][actor_critic_lambda][continuing_actor_critic]")
{
    // α^θ = 0: the 19-state random walk, learnt by TD(λ) and by the critic, on the same walks (to within 1e-12)
    for (double lambda : {0.0, 0.5, 0.9, 1.0}) {
        walk_type walk(19, -1, 1, 9);
        walk_types::policy td_memory, preferences;
        relearn::td_lambda<walk_types::state, walk_types::action> td(0.1, 0.9, lambda);
        relearn::actor_critic_lambda<walk_types::state, walk_types::action> critic(0.0, 0.1, 0.9, 0.5, lambda,
            [](const walk_types::state &) { return std::vector<walk_types::action> {walk_types::action(0)}; });
        for (int i = 0; i < 50; i++) {
            auto r = games::play(walk, [](const unsigned int &, const std::vector<unsigned int> &) { return 0u; }, 100000);
            td(r.episode, td_memory, r.terminated);
            critic(r.episode, preferences, r.terminated);
        }
        for (unsigned int s = 1; s <= 19; s++) {
            INFO("λ = " << lambda << ", state " << s);
            REQUIRE(critic.value(walk_types::state(s)) ==
                    Approx(td_memory.value(walk_types::state(s), walk_types::action(0))).epsilon(1e-12).margin(1e-15));
        }
    }
    // a task that never ends, with a single action: with λ^w = 0, the critic and R̄ are those of differential Sarsa,
    // whose one action makes it differential TD(0) (to within 1e-12)
    std::mt19937 prng(10);
    std::uniform_int_distribution<int> pick_state(0, 4);
    std::uniform_int_distribution<int> pick_reward(-1, 3);
    episode e;
    for (int i = 0; i < 2000; i++) {
        e.push_back({state(pick_reward(prng), "s" + std::to_string(pick_state(prng))), action("only")});
    }
    relearn::differential_sarsa<state,action> sarsa(0.1, 0.05);
    relearn::continuing_actor_critic<state,action> continuing(0.0, 0.1, 0.05, 0.5, 0.0,
        [](const state &) { return std::vector<action> {action("only")}; });
    policy sarsa_memory, preferences;
    sarsa(e, sarsa_memory);
    continuing(e, preferences);
    REQUIRE(continuing.rho == Approx(sarsa.rho).epsilon(1e-12));
    for (int k = 0; k < 5; k++) {
        const state s("s" + std::to_string(k));
        REQUIRE(continuing.value(s) == Approx(sarsa_memory.value(s, action("only"))).epsilon(1e-12).margin(1e-15));
    }
}

/// verifies the actor: with a critic that learns nothing and λ^θ = 1, it makes the changes of REINFORCE
TEST_CASE("actor_critic_lambda: with lambda = 1 and no critic, the actor is REINFORCE", "[policy_gradient][actor_critic_lambda]")
{
    // α^w = 0, so every δ is r_t, and z^θ = Σ_k γ^(t-k) γ^k ∇ ln π_k: the sum over t of α^θ r_t z^θ_t is
    // α^θ Σ_k γ^k G_k ∇ ln π_k, REINFORCE's. On episodes that never come back to a state, every ∇ ln π_k is worked out
    // from the same preferences in both (to within 1e-12)
    std::mt19937 prng(11);
    std::uniform_int_distribution<int> pick_action(0, 1);
    std::uniform_int_distribution<int> pick_reward(-2, 3);
    std::uniform_int_distribution<int> pick_length(1, 10);
    for (double gamma : {1.0, 0.8}) {
        relearn::reinforce<state,action> reinforce(0.2, gamma, left_right);
        relearn::actor_critic_lambda<state,action> actor(0.2, 0.0, gamma, 1.0, 0.5, left_right);
        policy reinforce_preferences, actor_preferences;
        for (int i = 0; i < 100; i++) {
            episode e;
            const int length = pick_length(prng);
            for (int k = 0; k < length; k++) {
                // each episode visits its own states, once each: s<episode>_<step>
                e.push_back({state(pick_reward(prng), "s" + std::to_string(i % 7) + "_" + std::to_string(k)),
                             action(pick_action(prng) == 0 ? "left" : "right")});
            }
            reinforce(e, reinforce_preferences);
            actor(e, actor_preferences);
        }
        for (int i = 0; i < 7; i++) {
            for (int k = 0; k < 10; k++) {
                const state s("s" + std::to_string(i) + "_" + std::to_string(k));
                for (const auto & a : {"left", "right"}) {
                    INFO("γ = " << gamma << ", " << s.trait() << ", " << a);
                    REQUIRE(actor_preferences.value(s, action(a)) ==
                            Approx(reinforce_preferences.value(s, action(a))).epsilon(1e-12).margin(1e-14));
                }
            }
        }
    }
}

/// verifies when an episode goes on from one call to the next, and when a new one starts
TEST_CASE("actor_critic_lambda: episodes cut short, and one call after another", "[policy_gradient][actor_critic_lambda]")
{
    auto x = step {state(1, "x"), action("right")};
    auto y = step {state(0, "y"), action("left")};
    auto z = step {state(2, "z"), action("right")};
    // an episode started with start_episode() and learnt a step at a time is the episode learnt at once
    relearn::actor_critic_lambda<state,action> stepwise(0.5, 0.5, 0.9, 0.8, 0.8, left_right);
    relearn::actor_critic_lambda<state,action> at_once(0.5, 0.5, 0.9, 0.8, 0.8, left_right);
    policy stepwise_preferences, at_once_preferences;
    stepwise.start_episode();
    stepwise(episode {x, {state(0, "y"), action("right")}}, stepwise_preferences, false);
    stepwise(episode {y, {state(0, "z"), action("left")}}, stepwise_preferences, false);
    stepwise(episode {z}, stepwise_preferences, true);
    at_once(episode {x, y, z}, at_once_preferences, true);
    for (const auto & link : {x, y, z}) {
        REQUIRE(stepwise.value(link.state) == at_once.value(link.state));
        for (const auto & a : {"left", "right"}) {
            REQUIRE(stepwise_preferences.value(link.state, action(a)) == at_once_preferences.value(link.state, action(a)));
        }
    }
    // after the end of an episode, the next call is a new one: I = 1 and every trace 0
    REQUIRE(stepwise.discount() == 1);
    stepwise(episode {z}, stepwise_preferences, true);
    REQUIRE(stepwise.critic_trace(state("x")) == Approx(0));
    REQUIRE(stepwise.actor_traces.trace(state("x"), action("right")) == Approx(0));
    // without start_episode(), the steps of an episode given one call at a time are learnt as episodes of their own,
    // even the next one after an episode cut short that starts with the very link it ended with
    relearn::actor_critic_lambda<state,action> unopened(0.5, 0.5, 0.9, 0.8, 0.8, left_right);
    policy unopened_preferences;
    unopened(episode {x, y}, unopened_preferences, false);
    REQUIRE(unopened.critic_trace(state("x")) == Approx(1));
    unopened(episode {y, z}, unopened_preferences, false);
    REQUIRE(unopened.critic_trace(state("x")) == Approx(0));
    REQUIRE(unopened.critic_trace(state("y")) == Approx(1));
    REQUIRE(unopened.discount() == Approx(0.9));
    // and start_episode() starts a new one whatever the last call was
    unopened.start_episode();
    REQUIRE(unopened.discount() == 1);
    REQUIRE(unopened.critic_trace(state("y")) == Approx(0));
    REQUIRE(unopened.actor_traces.entries().empty());
}

/// verifies the constructors of the Actor-Critics with traces, and brace initialisation
TEST_CASE("actor_critic_lambda and continuing_actor_critic: constructors and brace initialisation", "[policy_gradient]")
{
    relearn::actor_critic_lambda<state,action> episodic;
    REQUIRE(episodic.alpha_theta == Approx(0.01));
    REQUIRE(episodic.alpha_w == Approx(0.1));
    REQUIRE(episodic.gamma == Approx(0.9));
    REQUIRE(episodic.lambda_theta == Approx(0.9));
    REQUIRE(episodic.lambda_w == Approx(0.9));
    REQUIRE(episodic.discount() == 1);
    REQUIRE_FALSE(episodic.legal_actions);
    auto braced = relearn::actor_critic_lambda<state,action>{0.1, 0.2, 1.0, 0.3, 0.4, left_right};
    REQUIRE(braced.alpha_theta == Approx(0.1));
    REQUIRE(braced.alpha_w == Approx(0.2));
    REQUIRE(braced.gamma == Approx(1));
    REQUIRE(braced.lambda_theta == Approx(0.3));
    REQUIRE(braced.lambda_w == Approx(0.4));
    REQUIRE(braced.legal_actions);
    relearn::continuing_actor_critic<state,action> continuing;
    REQUIRE(continuing.alpha_theta == Approx(0.01));
    REQUIRE(continuing.alpha_w == Approx(0.1));
    REQUIRE(continuing.beta == Approx(0.01));
    REQUIRE(continuing.lambda_theta == Approx(0.9));
    REQUIRE(continuing.lambda_w == Approx(0.9));
    REQUIRE(continuing.rho == Approx(0));
    auto queue = relearn::continuing_actor_critic<state,action>{0.05, 0.2, 0.02, 0.5, 0.6, left_right};
    REQUIRE(queue.beta == Approx(0.02));
    REQUIRE(queue.lambda_w == Approx(0.6));
}

/// verifies that with traces, Actor-Critic learns the short corridor even when the cells look the same
TEST_CASE("actor_critic_lambda: the short corridor, with the cells looking the same", "[policy_gradient][actor_critic_lambda][short_corridor]")
{
    // where one-step Actor-Critic is misled by a critic that cannot tell the cells apart (the test above), the traces
    // carry each δ back to the actions before it, and the actor learns the stochastic policy p = 2 - √2 = 0.5858, worth -11.6.
    // The step sizes of Figure 13.2, α^θ = 2^-9 and α^w = 2^-6, γ = 1, λ^θ = λ^w = 0.9, 5 runs of 1000 episodes,
    // from π(right) = 0.05. Over 10 seeds (the same numbers with libstdc++ and with libc++), every run learnt
    // p = 0.547 to 0.674, worth -12.07 to -11.66 (against -1449 to -1074 for one-step Actor-Critic)
    corridor_lambda learner(std::pow(2.0, -9), std::pow(2.0, -6), 1.0, 0.9, 0.9, corridor_actions);
    auto learnt = learn_corridor(learner, false, 1000, 5, 1);
    for (std::size_t run = 0; run < learnt.right.size(); run++) {
        INFO("run " << run << ": p = " << learnt.right[run] << ", worth " << learnt.values[run]);
        REQUIRE(learnt.right[run] > 0.5);
        REQUIRE(learnt.right[run] < 0.72);
        REQUIRE(learnt.values[run] > -12.3);
    }
}

/// verifies that with traces, Actor-Critic still learns the short corridor when the cells are shown
TEST_CASE("actor_critic_lambda: the short corridor, with the cells shown", "[policy_gradient][actor_critic_lambda][short_corridor]")
{
    // the best policy is right, left, right, worth -3; α^θ = 2^-4, α^w = 2^-3, γ = 1, λ^θ = λ^w = 0.9,
    // 20 runs of 1000 episodes, from π(right) = 1/2 everywhere. Over 10 seeds (the same numbers with libstdc++
    // and with libc++), every run learnt a policy worth -3.040 or more, the median π(right) was 0.9908 to 0.9920
    // in the first cell, and the median value -3.026 to -3.024
    corridor_lambda learner(std::pow(2.0, -4), std::pow(2.0, -3), 1.0, 0.9, 0.9, corridor_actions);
    auto learnt = learn_corridor(learner, true, 1000, 20, 1);
    INFO("median π(right) " << median(learnt.right) << ", median value " << median(learnt.values));
    REQUIRE(above(learnt.values, -3.1) == learnt.values.size());
    REQUIRE(median(learnt.right) > 0.98);
}

/// verifies the continuing Actor-Critic on the access-control queue: its policy earns close to the optimal average reward
TEST_CASE("continuing_actor_critic: the access-control queue", "[policy_gradient][continuing_actor_critic][access_control]")
{
    // α^θ = 0.01, α^w = 0.1, β = 0.01, λ^θ = λ^w = 0.9, 1000000 steps learnt after every step, in pieces of 1000.
    // The optimal policy earns ρ* = 2.7476 per step (tests/solvers.hpp, checked in tests/r_learning_test.cpp).
    // The greedy policy of the preferences learnt (accepting on a tie) is worked out exactly from the rules.
    // Over 10 seeds, with libstdc++ and with libc++, the greedy policy earned 2.7343 to 2.7474, the agent earned
    // 2.7246 to 2.7521 whilst learning (second half), and the mean R̄ was 0.001 to 0.026 above it
    queue_type queue(10, 0.06, 1);
    std::mt19937 prng(1);
    relearn::continuing_actor_critic<queue_types::state, queue_types::action> learner(0.01, 0.1, 0.01, 0.9, 0.9, queue_actions);
    queue_types::policy preferences;
    auto agent = [&](const queue_type::state_type & s,
                     const std::vector<unsigned int> &) {
        return learner.sample(queue_types::state(s), preferences, prng).trait();
    };
    // over the second half, the reward earned per step, and R̄ (which moves a lot from one step to the next)
    double earned = 0;
    double mean_rho = 0;
    for (int piece = 0; piece < 1000; piece++) {
        auto r = games::play_online_actor_critic(queue, agent, learner, preferences, 1000);
        if (piece >= 500) {
            earned += r.total_reward / 500000;
            mean_rho += learner.rho / 500;
        }
    }
    auto greedy = [&](const queue_type::state_type & s) {
        if (s.free == 0) {
            return static_cast<unsigned int>(queue_type::reject);
        }
        const double accept = preferences.value(queue_types::state(s), queue_types::action(queue_type::accept));
        const double reject = preferences.value(queue_types::state(s), queue_types::action(queue_type::reject));
        return static_cast<unsigned int>(accept >= reject ? queue_type::accept : queue_type::reject);
    };
    queue_type rules;
    const double greedy_earns = rules.average_reward(greedy);
    INFO("earned whilst learning (second half) " << earned << ", mean R̄ " << mean_rho << ", the greedy policy earns " << greedy_earns);
    REQUIRE(greedy_earns > 2.7476 - 0.03);
    REQUIRE(earned > 2.7476 - 0.05);
    REQUIRE(std::fabs(mean_rho - earned) < 0.05);
}
