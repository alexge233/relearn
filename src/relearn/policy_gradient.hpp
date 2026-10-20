#ifndef RELEARN_POLICY_GRADIENT_HPP
#define RELEARN_POLICY_GRADIENT_HPP
/**
 * Copyright 2015 Alexander Giokas
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * #http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file relearn/policy_gradient.hpp
 * @brief policy gradient methods: the softmax policy (`softmax_policy`), REINFORCE (`reinforce`),
 *        REINFORCE with a baseline (`reinforce_with_baseline`) and one-step Actor-Critic (`actor_critic`)
 */
#include "core.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <functional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace relearn
{
/**
 * @struct softmax_policy This is the softmax policy over action preferences (Sutton & Barto, Section 13.1)
 * @brief the policy π(a | s) itself, worked out from a preference h(s, a) for every legal action
 * @param legal_actions gives the legal actions of a state - required
 *
 * The policy gradient methods learn the policy itself, rather than the values of the actions.
 * Their parameters θ are a *preference* h(s, a) for each state and legal action, kept in a
 * `relearn::policy` (a preference never learnt is 0), and the policy is their softmax
 * (equation 13.2):
 *
 * π(a | s) = e^h(s,a) / Σ_b e^h(s,b), over the legal actions b of s
 *
 * The higher the preference of an action, compared with the others, the more often it is taken,
 * but every action keeps some chance, so the policy can stay stochastic (and it explores).
 *
 * Gradient ascent needs ∇ ln π(a | s), the direction in which to move the preferences so that
 * a becomes more likely in s. With respect to the preference h(s, b) of each legal action b, it is
 *
 * ∂ ln π(a | s) / ∂ h(s, b) = 1[a = b] - π(b | s)
 *
 * (the preferences of the other states do not change π(a | s), so their part is 0).
 * This is equation 13.9 of Exercise 13.3, with the tabular features x(s, b) = 1 for the
 * pair (s, b) and 0 for the others.
 *
 * The learners below are softmax policies too, so an agent can simply ask its learner
 * for π(a | s) (`probability`) or for an action drawn from it (`sample`).
 */
template <class state_class,
          class action_class,
          typename value_type = double>
struct softmax_policy
{
    /// the legal actions of a state, each with its probability π(a | s), or its ∇ ln π
    using distribution = std::vector<std::pair<action_class,value_type>>;
    /// the legal actions of a state - required
    std::function<std::vector<action_class>(const state_class &)> legal_actions;

    /// ctor with the legal actions @param legal of each state
    softmax_policy(std::function<std::vector<action_class>(const state_class &)> legal = nullptr);

    /**
     * @return every legal action of @param s_t with its probability π(a | s_t),
     *         from the preferences h(s, a) in @param preferences
     */
    distribution probabilities(const state_class & s_t,
                               const policy<state_class,action_class,value_type> & preferences) const;
    /// @return π(@param a_t | @param s_t), from the preferences h(s, a) in @param preferences
    value_type probability(const state_class & s_t,
                           const action_class & a_t,
                           const policy<state_class,action_class,value_type> & preferences) const;
    /**
     * @return an action drawn from π(· | @param s_t), from the preferences in @param preferences
     * @param prng draws the random number
     */
    action_class sample(const state_class & s_t,
                        const policy<state_class,action_class,value_type> & preferences,
                        std::mt19937 & prng) const;
    /**
     * @return ∇ ln π(@param a_t | @param s_t): every legal action b of @param s_t with
     *         1[a_t = b] - π(b | s_t), the derivative of ln π(a_t | s_t) by h(s_t, b)
     */
    distribution log_gradient(const state_class & s_t,
                              const action_class & a_t,
                              const policy<state_class,action_class,value_type> & preferences) const;
    /**
     * @brief move the preferences of @param s_t by @param size along ∇ ln π(@param a_t | @param s_t):
     *        h(s_t, b) = h(s_t, b) + size * (1[a_t = b] - π(b | s_t)) for every legal action b
     *        (all of them worked out from the preferences before the step) - @param preferences will be modified
     */
    void ascend(const state_class & s_t,
                const action_class & a_t,
                value_type size,
                policy<state_class,action_class,value_type> & preferences) const;
};

/**
 * @struct reinforce This is REINFORCE, Monte Carlo policy gradient (Sutton & Barto, Section 13.3)
 * @brief REINFORCE learns the policy itself from whole episodes: each action taken becomes more
 *        likely in proportion to the return that followed it
 * @param alpha is the step size α
 * @param gamma is the discount rate
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t (the book's R_{t+1}).
 *
 * The episode must be played by the policy being learnt: actions drawn with `sample`,
 * from the same preferences. Once it has ended, for each step t = 0, 1, ..., T-1 in turn:
 *
 * G_t = r_t + γ r_{t+1} + ... + γ^{T-1-t} r_{T-1}
 * θ = θ + α γ^t G_t ∇ ln π(a_t | s_t, θ)
 *
 * where θ are the preferences h(s, a) (have a look at `softmax_policy`). The γ^t factor is in
 * the book's pseudocode: the quantity REINFORCE climbs is the value of the *start* state,
 * v_π(s_0), and a step taken at time t only counts towards it discounted by γ^t, so the later
 * steps of an episode move θ less. With γ = 1 it is simply 1.
 *
 * As in the book's pseudocode, θ changes from one step to the next whilst the episode is
 * walked, so the ∇ ln π of each step is worked out from the preferences left by the step before.
 *
 * REINFORCE needs the real return, so it only learns from episodes that ended:
 * an episode cut short (`terminated` false) is not learnt at all.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct reinforce : public softmax_policy<state_class,action_class,value_type>
{
    /// the step size α - you may change this as you process episodes
    value_type alpha = 0.01;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;

    /**
     * @brief ctor with the step size @param initial_alpha (0.01 by default), the discount rate
     *        @param initial_gamma (0.9 by default), and the legal actions @param legal of each state
     */
    reinforce(value_type initial_alpha = 0.01,
              value_type initial_gamma = 0.9,
              std::function<std::vector<action_class>(const state_class &)> legal = nullptr);

    /**
     * @brief learn from an episode played by π - @param preferences (θ) will be modified
     * @param terminated true if the episode really ended after its last link;
     *        an episode cut short is not learnt, since its return is not known
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & preferences,
                    bool terminated = true);
};

/**
 * @struct reinforce_with_baseline This is REINFORCE with a baseline (Sutton & Barto, Section 13.4)
 * @brief REINFORCE, where each return is compared with the value v(s) learnt for its state
 * @param alpha_theta is the step size α^θ of the policy
 * @param alpha_w is the step size α^w of the state values (the baseline)
 * @param gamma is the discount rate
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t (the book's R_{t+1}).
 *
 * In REINFORCE, every action becomes more likely in proportion to its return, so when all
 * the returns are negative (as in the short corridor) every action taken is pushed down,
 * and only the size of the push tells the good ones from the bad ones. The baseline is the
 * value of the state, v(s), learnt alongside the policy (here a table: one value per state),
 * and the push is the return *compared with* what was expected in that state. Once the episode
 * has ended, for each step t = 0, 1, ..., T-1 in turn:
 *
 * G_t = r_t + γ r_{t+1} + ... + γ^{T-1-t} r_{T-1}
 * δ = G_t - v(s_t)
 * v(s_t) = v(s_t) + α^w δ                      (∇v̂(s_t, w) is 1 for the value of s_t)
 * θ = θ + α^θ γ^t δ ∇ ln π(a_t | s_t, θ)
 *
 * The γ^t factor is the one of `reinforce`. The baseline does not depend on the action, so it
 * leaves the direction of the update right on average, but it makes it far less noisy.
 * The state values start at 0 and are kept inside the learner (have a look at `value`).
 * Like `reinforce`, it only learns from episodes that ended.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct reinforce_with_baseline : public softmax_policy<state_class,action_class,value_type>
{
    /// the step size α^θ of the policy - you may change this as you process episodes
    value_type alpha_theta = 0.01;
    /// the step size α^w of the state values - you may change this as you process episodes
    value_type alpha_w = 0.1;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;

    /**
     * @brief ctor with the step sizes @param initial_alpha_theta (α^θ, 0.01 by default) and
     *        @param initial_alpha_w (α^w, 0.1 by default), the discount rate @param initial_gamma
     *        (0.9 by default), and the legal actions @param legal of each state
     */
    reinforce_with_baseline(value_type initial_alpha_theta = 0.01,
                            value_type initial_alpha_w = 0.1,
                            value_type initial_gamma = 0.9,
                            std::function<std::vector<action_class>(const state_class &)> legal = nullptr);

    /**
     * @brief learn from an episode played by π - @param preferences (θ) and the state values will be modified
     * @param terminated true if the episode really ended after its last link;
     *        an episode cut short is not learnt, since its return is not known
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & preferences,
                    bool terminated = true);

    /// @return the value v(@param s_t) learnt for the baseline, 0 if the state was never learnt
    value_type value(const state_class & s_t) const;
private:
    // the state values v(s), the baseline: [state] => value
    std::unordered_map<state_class,
                       value_type,
                       hasher<state_class>> values_;
};

/**
 * @struct actor_critic This is one-step Actor-Critic, episodic (Sutton & Barto, Section 13.5)
 * @brief the policy (the actor) learns after every step, from the TD error of the state values (the critic)
 * @param alpha_theta is the step size α^θ of the policy
 * @param alpha_w is the step size α^w of the state values (the critic)
 * @param gamma is the discount rate
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t (the book's R_{t+1}).
 *
 * REINFORCE waits for the end of the episode, for the real return. Actor-Critic replaces the
 * return with a one-step estimate, as TD(0) does: the reward, plus the discounted value of
 * the state reached. The critic learns the state values v(s) (here a table), and its TD error
 * tells the actor whether the action just taken turned out better or worse than expected.
 * After each step (s_t, a_t, r_t, s_{t+1}):
 *
 * δ = r_t + γ v(s_{t+1}) - v(s_t)              (v(s_{t+1}) = 0 when the episode ended)
 * v(s_t) = v(s_t) + α^w δ                      (∇v̂(s_t, w) is 1 for the value of s_t)
 * θ = θ + α^θ I δ ∇ ln π(a_t | s_t, θ)
 * I = γ I
 *
 * where I = γ^t, the γ^t factor of `reinforce`, is 1 at the start of each episode.
 *
 * Basically, it learns as the episode is played, and works for episodes that are long or never end,
 * but δ is only as good as the critic's values: when the critic cannot tell states apart,
 * δ is biased, and the actor may learn a bad policy (as it does in the short corridor).
 *
 * Learn after every step, as SARSA does (have a look at `games::play_online_actor_critic`):
 * each link is learnt from the state of the link after it, and like `sarsa`, an episode cut
 * short (`terminated` false) only gives its last link as the state reached. The learner
 * keeps I from one call to the next, multiplies it by γ after each step learnt, and puts it
 * back to 1 once the last step of an episode that ended is learnt. A whole episode can be
 * learnt in one call too.
 * @warning after an episode cut short, call `start_episode` before the next one, so that I is 1 again
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct actor_critic : public softmax_policy<state_class,action_class,value_type>
{
    /// the step size α^θ of the policy (the actor) - you may change this as you process episodes
    value_type alpha_theta = 0.01;
    /// the step size α^w of the state values (the critic) - you may change this as you process episodes
    value_type alpha_w = 0.1;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;

    /**
     * @brief ctor with the step sizes @param initial_alpha_theta (α^θ, 0.01 by default) and
     *        @param initial_alpha_w (α^w, 0.1 by default), the discount rate @param initial_gamma
     *        (0.9 by default), and the legal actions @param legal of each state
     */
    actor_critic(value_type initial_alpha_theta = 0.01,
                 value_type initial_alpha_w = 0.1,
                 value_type initial_gamma = 0.9,
                 std::function<std::vector<action_class>(const state_class &)> legal = nullptr);

    /**
     * @brief learn from the steps of an episode - @param preferences (θ) and the state values will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short, or for the steps of an episode still going on:
     *        its last link is then not learnt, and only serves as the state reached by the link before it.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & preferences,
                    bool terminated = true);

    /// @brief start a new episode: I = 1
    void start_episode();
    /// @return I, the γ^t of the next step to learn (1 at the start of an episode)
    value_type discount() const;
    /// @return the value v(@param s_t) learnt by the critic, 0 if the state was never learnt
    value_type value(const state_class & s_t) const;
private:
    // I = γ^t, for the step t of the episode learnt next
    value_type discount_ = 1;
    // the state values v(s), the critic: [state] => value
    std::unordered_map<state_class,
                       value_type,
                       hasher<state_class>> values_;
};

/********************************************************************************
 ***********               Template Implementations
 ********************************************************************************/

template <class state_class,
          class action_class,
          typename value_type>
softmax_policy<state_class,action_class,value_type
              >::softmax_policy(std::function<std::vector<action_class>(const state_class &)> legal)
: legal_actions(legal)
{}

template <class state_class,
          class action_class,
          typename value_type>
typename softmax_policy<state_class,action_class,value_type>::distribution
    softmax_policy<state_class,action_class,value_type
                  >::probabilities(const state_class & s_t,
                                   const policy<state_class,action_class,value_type> & preferences) const
{
    assert(legal_actions && "softmax_policy needs the legal actions of each state");
    auto legal = legal_actions(s_t);
    distribution pi;
    if (legal.empty()) {
        return pi;
    }
    // the highest preference: subtracting it from every preference changes nothing in π,
    // but keeps e^h from overflowing (the largest power is then e^0 = 1)
    value_type highest = preferences.value(s_t, legal.front());
    for (const auto & b : legal) {
        highest = std::max(highest, preferences.value(s_t, b));
    }
    // e^(h(s,b) - highest) for every legal action b, and their sum
    value_type sum = 0;
    for (const auto & b : legal) {
        value_type power = std::exp(preferences.value(s_t, b) - highest);
        pi.push_back(std::make_pair(b, power));
        sum += power;
    }
    // π(b | s) = e^h(s,b) / Σ_c e^h(s,c)   (Sutton & Barto, equation 13.2)
    for (auto & b : pi) {
        b.second = b.second / sum;
    }
    return pi;
}

template <class state_class,
          class action_class,
          typename value_type>
value_type softmax_policy<state_class,action_class,value_type
                         >::probability(const state_class & s_t,
                                        const action_class & a_t,
                                        const policy<state_class,action_class,value_type> & preferences) const
{
    // find a_t amongst the legal actions; an action that is not legal is never taken
    for (const auto & b : probabilities(s_t, preferences)) {
        if (b.first == a_t) {
            return b.second;
        }
    }
    return 0;
}

template <class state_class,
          class action_class,
          typename value_type>
action_class softmax_policy<state_class,action_class,value_type
                           >::sample(const state_class & s_t,
                                     const policy<state_class,action_class,value_type> & preferences,
                                     std::mt19937 & prng) const
{
    auto pi = probabilities(s_t, preferences);
    assert(!pi.empty() && "a state with no legal action has nothing to sample");
    // draw u in [0, 1), and take the first action whose cumulative probability passes it
    std::uniform_real_distribution<value_type> uniform(0, 1);
    const value_type u = uniform(prng);
    value_type cumulative = 0;
    for (const auto & b : pi) {
        cumulative += b.second;
        if (u < cumulative) {
            return b.first;
        }
    }
    // the probabilities may add up to a hair under 1: then u fell in the last action's share
    return pi.back().first;
}

template <class state_class,
          class action_class,
          typename value_type>
typename softmax_policy<state_class,action_class,value_type>::distribution
    softmax_policy<state_class,action_class,value_type
                  >::log_gradient(const state_class & s_t,
                                  const action_class & a_t,
                                  const policy<state_class,action_class,value_type> & preferences) const
{
    // ∂ ln π(a_t | s_t) / ∂ h(s_t, b) = 1[a_t = b] - π(b | s_t)   (Sutton & Barto, equation 13.9)
    auto gradient = probabilities(s_t, preferences);
    for (auto & b : gradient) {
        b.second = (b.first == a_t ? value_type(1) : value_type(0)) - b.second;
    }
    return gradient;
}

template <class state_class,
          class action_class,
          typename value_type>
void softmax_policy<state_class,action_class,value_type
                   >::ascend(const state_class & s_t,
                             const action_class & a_t,
                             value_type size,
                             policy<state_class,action_class,value_type> & preferences) const
{
    // ∇ ln π(a_t | s_t), worked out in full before any preference changes
    auto gradient = log_gradient(s_t, a_t, preferences);
    // h(s_t, b) = h(s_t, b) + size * (1[a_t = b] - π(b | s_t))
    for (const auto & b : gradient) {
        auto h = preferences.value(s_t, b.first);
        preferences.update(s_t, b.first, h + size * b.second);
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
reinforce<state_class,action_class,markov_chain,value_type
         >::reinforce(value_type initial_alpha,
                      value_type initial_gamma,
                      std::function<std::vector<action_class>(const state_class &)> legal)
: softmax_policy<state_class,action_class,value_type>(legal),
  alpha(initial_alpha),
  gamma(initial_gamma)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void reinforce<state_class,action_class,markov_chain,value_type
              >::operator()(markov_chain episode,
                            policy<state_class,action_class,value_type> & preferences,
                            bool terminated)
{
    // an episode cut short has no return yet: nothing to learn
    if (!terminated || episode.size() == 0) {
        return;
    }
    // the return that followed each step, worked out backwards: G_t = r_t + γ G_{t+1}
    std::vector<value_type> G(episode.size());
    value_type following = 0;
    for (std::size_t i = episode.size(); i-- > 0; ) {
        following = episode[i].state.reward() + gamma * following;
        G[i] = following;
    }
    // then each step in turn, from the first: θ = θ + α γ^t G_t ∇ ln π(a_t | s_t, θ)
    value_type discount = 1;
    for (std::size_t t = 0; t < episode.size(); t++) {
        this->ascend(episode[t].state, episode[t].action, alpha * discount * G[t], preferences);
        // γ^t for the next step
        discount = discount * gamma;
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
reinforce_with_baseline<state_class,action_class,markov_chain,value_type
                       >::reinforce_with_baseline(value_type initial_alpha_theta,
                                                  value_type initial_alpha_w,
                                                  value_type initial_gamma,
                                                  std::function<std::vector<action_class>(const state_class &)> legal)
: softmax_policy<state_class,action_class,value_type>(legal),
  alpha_theta(initial_alpha_theta),
  alpha_w(initial_alpha_w),
  gamma(initial_gamma)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void reinforce_with_baseline<state_class,action_class,markov_chain,value_type
                            >::operator()(markov_chain episode,
                                          policy<state_class,action_class,value_type> & preferences,
                                          bool terminated)
{
    // an episode cut short has no return yet: nothing to learn
    if (!terminated || episode.size() == 0) {
        return;
    }
    // the return that followed each step, worked out backwards: G_t = r_t + γ G_{t+1}
    std::vector<value_type> G(episode.size());
    value_type following = 0;
    for (std::size_t i = episode.size(); i-- > 0; ) {
        following = episode[i].state.reward() + gamma * following;
        G[i] = following;
    }
    // then each step in turn, from the first
    value_type discount = 1;
    for (std::size_t t = 0; t < episode.size(); t++) {
        const auto & s_t = episode[t].state;
        // δ = G_t - v(s_t): how much better the return was than expected
        const value_type v_t = value(s_t);
        const value_type delta = G[t] - v_t;
        // v(s_t) = v(s_t) + α^w δ
        values_[s_t] = v_t + alpha_w * delta;
        // θ = θ + α^θ γ^t δ ∇ ln π(a_t | s_t, θ)
        this->ascend(s_t, episode[t].action, alpha_theta * discount * delta, preferences);
        // γ^t for the next step
        discount = discount * gamma;
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type reinforce_with_baseline<state_class,action_class,markov_chain,value_type
                                  >::value(const state_class & s_t) const
{
    auto found = values_.find(s_t);
    return found == values_.end() ? value_type(0) : found->second;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
actor_critic<state_class,action_class,markov_chain,value_type
            >::actor_critic(value_type initial_alpha_theta,
                            value_type initial_alpha_w,
                            value_type initial_gamma,
                            std::function<std::vector<action_class>(const state_class &)> legal)
: softmax_policy<state_class,action_class,value_type>(legal),
  alpha_theta(initial_alpha_theta),
  alpha_w(initial_alpha_w),
  gamma(initial_gamma)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void actor_critic<state_class,action_class,markov_chain,value_type
                 >::operator()(markov_chain episode,
                               policy<state_class,action_class,value_type> & preferences,
                               bool terminated)
{
    // a cut-short episode does not learn its last link: it is only the state reached
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size()
                                                               : episode.size() - 1;
    for (std::size_t i = 0; i < learnt; i++) {
        const auto & s_t = episode[i].state;
        // the last link of an episode that ended has no next state: v(s_{t+1}) = 0
        const bool last = terminated && i == episode.size() - 1;
        const value_type v_next = last ? value_type(0) : value(episode[i + 1].state);
        // δ = r_t + γ v(s_{t+1}) - v(s_t): the TD error of the critic
        const value_type v_t = value(s_t);
        const value_type delta = s_t.reward() + gamma * v_next - v_t;
        // the critic: v(s_t) = v(s_t) + α^w δ
        values_[s_t] = v_t + alpha_w * delta;
        // the actor: θ = θ + α^θ I δ ∇ ln π(a_t | s_t, θ)
        this->ascend(s_t, episode[i].action, alpha_theta * discount_ * delta, preferences);
        // I = γ I, or 1 again once the episode has ended
        discount_ = last ? value_type(1) : gamma * discount_;
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void actor_critic<state_class,action_class,markov_chain,value_type>::start_episode()
{
    discount_ = 1;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type actor_critic<state_class,action_class,markov_chain,value_type>::discount() const
{
    return discount_;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type actor_critic<state_class,action_class,markov_chain,value_type
                       >::value(const state_class & s_t) const
{
    auto found = values_.find(s_t);
    return found == values_.end() ? value_type(0) : found->second;
}
} // end of namespace
#endif
