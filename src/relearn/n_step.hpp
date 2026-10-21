#ifndef RELEARN_N_STEP_HPP
#define RELEARN_N_STEP_HPP
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
 * @file relearn/n_step.hpp
 * @brief n-step SARSA (`n_step_sarsa`), and n-step TD prediction (`n_step_td`), which is TD(0) when n = 1;
 *        n-step Expected SARSA (`n_step_expected_sarsa`), off-policy n-step SARSA (`off_policy_n_step_sarsa`),
 *        n-step tree backup (`n_step_tree_backup`) and n-step Q(σ) (`n_step_q_sigma`)
 */
#include "core.hpp"
#include "schedules.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
#include <utility>
#include <vector>

namespace relearn
{
/**
 * @struct n_step_sarsa This is n-step SARSA (Sutton & Barto, Section 7.2)
 * @brief n-step SARSA learns each value from the next n rewards, and the value of the pair reached after them
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param n is the number of rewards each update looks ahead
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * The n-step return of step τ adds up the next n rewards, and then the value of
 * the state/action pair the agent reached after them (Section 7.2, equation 7.4):
 *
 * G_{τ:τ+n} = r_τ + γ * r_{τ+1} + ... + γ^{n-1} * r_{τ+n-1} + γ^n * Q(s_{τ+n}, a_{τ+n})
 *
 * If the episode ends before that, at T (the number of steps), it is simply the
 * return to the end, r_τ + ... + γ^{T-τ-1} * r_{T-1}, with nothing after it. Then
 *
 * Q(s_τ,a_τ) = Q(s_τ,a_τ) + α * (G_{τ:τ+n} - Q(s_τ,a_τ))
 *
 * for τ = 0, 1, ..., T-1, in that order, each update using the values as they are
 * at that point, exactly as the book's algorithm does whilst the episode is played.
 * With n = 1 this is `sarsa`; with n ≥ T, constant-α Monte Carlo.
 *
 * An episode cut short (`terminated` false) ends with the state reached and the action
 * chosen next, which is not a step. A step is learnt only once its n rewards are all
 * known, so the last n-1 steps of an episode cut short are not learnt: this is where
 * the book's algorithm would be, if it were stopped there. The same rule lets the
 * learner learn whilst the episode is played: give it the last n steps and the next
 * link after every step, as `games::play_online(..., n)` does, and it learns the oldest of them.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct n_step_sarsa
{
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// n: the number of rewards each update looks ahead, at least 1 (1 is SARSA, or TD(0))
    unsigned int n = 1;
    /**
     * optional: the learning rate of each update, given the number of times k = 1, 2, ...
     * the state/action pair has been updated, e.g. `relearn::polynomial_decay<>()` for α = 1/k.
     * If it is not set, `alpha` is used for every update.
     */
    std::function<value_type(std::size_t)> learning_rate;
    /// the number of updates of each state/action pair (counted only when `learning_rate` is set)
    visit_counter<state_class,action_class> visits;

    /**
     * @brief ctor with a learning rate @param initial_alpha and a discount rate @param initial_gamma,
     *        both 0.9 by default, and the number of steps @param initial_n (1 by default)
     */
    n_step_sarsa(value_type initial_alpha = 0.9,
                 value_type initial_gamma = 0.9,
                 unsigned int initial_n = 1);

    /**
     * @return G_{τ:τ+n}, the n-step return of the step at @param tau in @param episode,
     *         using the values of @param policy_map after the n rewards, if the episode goes on that far
     */
    value_type n_step_return(const markov_chain & episode,
                             std::size_t tau,
                             const policy<state_class,action_class,value_type> & policy_map) const;

    /**
     * @brief learn from an episode - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short (e.g., by a step limit), or for the
     *        last n steps of an episode still being played: its last link is then the
     *        state reached and the action chosen next, and only the steps whose n
     *        rewards are all in the episode are learnt.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);
};

/**
 * @brief n-step TD prediction (Sutton & Barto, Section 7.1), and TD(0) when n = 1 (Section 6.1)
 *
 * Prediction learns how much each state is worth to an agent whose policy is fixed. relearn
 * keeps values for state/action pairs, so a prediction task has a single action per state
 * (as `games::random_walk` does), and V(s) is the value of (s, that action). The update is then
 *
 * V(s_τ) = V(s_τ) + α * (r_τ + ... + γ^{n-1} * r_{τ+n-1} + γ^n * V(s_{τ+n}) - V(s_τ))
 *
 * which is exactly n-step SARSA's, so `n_step_td` is `n_step_sarsa` under the name of prediction.
 * With several actions per state it learns q_π, the value of each pair under the policy that played.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
using n_step_td = n_step_sarsa<state_class,action_class,markov_chain,value_type>;

/**
 * @brief the target policy π of the n-step learners below, as each action with its probability
 * @param probabilities is π(a | s), if it is set
 * @param s_t is the state
 * @param policy_map holds the values learnt so far
 * @return π(· | s_t), from @param probabilities if it is set; otherwise the greedy policy on
 *         the actions stored for s_t (those of highest value share the probability equally),
 *         and nothing if s_t is unknown
 */
template <class state_class,
          class action_class,
          typename value_type>
std::vector<std::pair<action_class,value_type>>
    target_policy(const std::function<std::vector<std::pair<action_class,value_type>>(const state_class &)> & probabilities,
                  const state_class & s_t,
                  const policy<state_class,action_class,value_type> & policy_map);

/// @return the probability of @param a_t in the distribution @param pi, or 0 if it is not in it
template <class action_class,
          typename value_type>
value_type probability_of(const std::vector<std::pair<action_class,value_type>> & pi,
                          const action_class & a_t);

/// @return V̄(s_t) = Σ_a π(a | s_t) Q(s_t, a), with π the distribution @param pi and Q the values of @param policy_map
template <class state_class,
          class action_class,
          typename value_type>
value_type expected_value_of(const std::vector<std::pair<action_class,value_type>> & pi,
                             const state_class & s_t,
                             const policy<state_class,action_class,value_type> & policy_map);

/**
 * @return the next n rewards of the step at @param tau in @param episode, discounted:
 *         r_τ + γ * r_{τ+1} + ... + γ^{n-1} * r_{τ+n-1}, or up to the end of the episode if it is nearer,
 *         and γ to the power of the number of rewards in @param discount
 */
template <typename markov_chain,
          typename value_type>
value_type discounted_rewards(const markov_chain & episode,
                              std::size_t tau,
                              unsigned int n,
                              value_type gamma,
                              value_type & discount);

/**
 * @struct n_step_expected_sarsa This is n-step Expected SARSA (Sutton & Barto, Sections 7.2 and 7.3)
 * @brief n-step Expected SARSA learns each value from the next n rewards, and the value the state
 *        reached after them is *expected* to have under the policy π
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param n is the number of rewards each update looks ahead
 * @param probabilities is the policy π being learnt
 * @param behaviour is the policy b that played the episodes, if it is not π (off-policy)
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * Where n-step SARSA ends its return with the value of the pair the agent reached,
 * n-step Expected SARSA ends it with the value of the state it reached, expected under π:
 *
 * G_{τ:τ+n} = r_τ + γ * r_{τ+1} + ... + γ^{n-1} * r_{τ+n-1} + γ^n * V̄(s_{τ+n}),    V̄(s) = Σ_a π(a | s) Q(s, a)
 *
 * (and simply the return to the end, if the episode ends sooner). Then
 *
 * Q(s_τ,a_τ) = Q(s_τ,a_τ) + α * ρ * (G_{τ:τ+n} - Q(s_τ,a_τ))
 *
 * On-policy (`behaviour` not set), ρ = 1. Off-policy, the episodes are played by another
 * policy b, and each step is weighted by how much more (or less) likely π was to take the
 * actions after a_τ than b was: the importance sampling ratio of Section 7.3,
 *
 * ρ = ρ_{τ+1:τ+n-1} = Π_{k=τ+1}^{min(τ+n-1, T-1)} π(a_k | s_k) / b(a_k | s_k)
 *
 * which has one factor less than off-policy n-step SARSA's: the action taken in s_{τ+n}
 * needs no correction, since the return uses all of its actions, as π weighs them.
 * With n = 1 this is `expected_sarsa`.
 *
 * As in `expected_sarsa`, `probabilities(s)` returns each action π may take in state s, with
 * its probability; if it is not set, π is the greedy policy on the values learnt. `behaviour(s, a)`
 * returns b(a | s), the probability that the policy which played the episode took a in s.
 * The probabilities are worked out when the step is learnt: if π or b change as values are learnt
 * (e.g. ε-greedy on them), they are those of that moment, as in the book's algorithm.
 * Episodes cut short and learning online work as for `n_step_sarsa`.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct n_step_expected_sarsa
{
    /// the actions the agent may take in a state, each with the probability of taking it
    using distribution = std::vector<std::pair<action_class,value_type>>;
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// n: the number of rewards each update looks ahead, at least 1 (1 is Expected SARSA)
    unsigned int n = 1;
    /**
     * optional: the learning rate of each update, given the number of times k = 1, 2, ...
     * the state/action pair has been updated, e.g. `relearn::polynomial_decay<>()` for α = 1/k.
     * If it is not set, `alpha` is used for every update.
     */
    std::function<value_type(std::size_t)> learning_rate;
    /// the number of updates of each state/action pair (counted only when `learning_rate` is set)
    visit_counter<state_class,action_class> visits;
    /// the policy π(a | s) being learnt - if not set, the greedy policy is used
    std::function<distribution(const state_class &)> probabilities;
    /// optional: b(a | s), the probability that the policy which played the episodes takes a in s (off-policy)
    std::function<value_type(const state_class &, const action_class &)> behaviour;

    /**
     * @brief ctor with a learning rate @param initial_alpha and a discount rate @param initial_gamma,
     *        both 0.9 by default, the number of steps @param initial_n (1 by default),
     *        and the policy @param pi (`probabilities`), not set by default
     */
    n_step_expected_sarsa(value_type initial_alpha = 0.9,
                          value_type initial_gamma = 0.9,
                          unsigned int initial_n = 1,
                          std::function<distribution(const state_class &)> pi = nullptr);

    /// @return V̄(s_t) = Σ_a π(a | s_t) Q(s_t, a), using the values of @param policy_map
    value_type expected_value(const state_class & s_t,
                              const policy<state_class,action_class,value_type> & policy_map) const;

    /**
     * @return G_{τ:τ+n}, the n-step return of the step at @param tau in @param episode,
     *         ending with the expected value of the state reached after the n rewards, if the episode goes on that far
     */
    value_type n_step_return(const markov_chain & episode,
                             std::size_t tau,
                             const policy<state_class,action_class,value_type> & policy_map) const;

    /// @return ρ_{τ+1:τ+n-1} for the step at @param tau in @param episode, or 1 if `behaviour` is not set
    value_type importance_ratio(const markov_chain & episode,
                                std::size_t tau,
                                const policy<state_class,action_class,value_type> & policy_map) const;

    /**
     * @brief learn from an episode - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link (see `n_step_sarsa`)
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);
};

/**
 * @struct off_policy_n_step_sarsa This is off-policy n-step SARSA (Sutton & Barto, Section 7.3)
 * @brief n-step SARSA that learns the values of a policy π from episodes played by another policy b
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param n is the number of rewards each update looks ahead
 * @param probabilities is the policy π being learnt
 * @param behaviour is the policy b that played the episodes
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * The return is n-step SARSA's (equation 7.4), from actions that b took. To learn π's values
 * from them, each update is weighted by the importance sampling ratio (equations 7.10 and 7.11):
 *
 * ρ_{τ+1:τ+n} = Π_{k=τ+1}^{min(τ+n, T-1)} π(a_k | s_k) / b(a_k | s_k)
 *
 * Q(s_τ,a_τ) = Q(s_τ,a_τ) + α * ρ_{τ+1:τ+n} * (G_{τ:τ+n} - Q(s_τ,a_τ))
 *
 * The first action, a_τ, is not in the ratio: Q(s_τ,a_τ) is the value of taking it, whoever chose it.
 * If π would never have taken one of the later actions, ρ is 0 and the step teaches nothing
 * about π (it is then not counted as an update of the pair); if b rarely takes the actions π
 * prefers, ρ can be large, and the updates vary a lot (have a look at docs/n_step.md).
 * With b = π, ρ = 1 and this is `n_step_sarsa`.
 *
 * `probabilities` and `behaviour` are as in `n_step_expected_sarsa`; `behaviour` is required.
 * Episodes cut short and learning online work as for `n_step_sarsa`.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct off_policy_n_step_sarsa
{
    /// the actions the agent may take in a state, each with the probability of taking it
    using distribution = std::vector<std::pair<action_class,value_type>>;
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// n: the number of rewards each update looks ahead, at least 1
    unsigned int n = 1;
    /**
     * optional: the learning rate of each update, given the number of times k = 1, 2, ...
     * the state/action pair has been updated, e.g. `relearn::polynomial_decay<>()` for α = 1/k.
     * If it is not set, `alpha` is used for every update.
     */
    std::function<value_type(std::size_t)> learning_rate;
    /// the number of updates of each state/action pair (counted only when `learning_rate` is set)
    visit_counter<state_class,action_class> visits;
    /// the policy π(a | s) being learnt - if not set, the greedy policy is used
    std::function<distribution(const state_class &)> probabilities;
    /// b(a | s): the probability that the policy which played the episodes takes a in s - required
    std::function<value_type(const state_class &, const action_class &)> behaviour;

    /**
     * @brief ctor with a learning rate @param initial_alpha and a discount rate @param initial_gamma,
     *        both 0.9 by default, the number of steps @param initial_n (1 by default),
     *        the policy being learnt @param pi (greedy if not set) and the behaviour policy @param b
     */
    off_policy_n_step_sarsa(value_type initial_alpha = 0.9,
                            value_type initial_gamma = 0.9,
                            unsigned int initial_n = 1,
                            std::function<distribution(const state_class &)> pi = nullptr,
                            std::function<value_type(const state_class &, const action_class &)> b = nullptr);

    /// @return G_{τ:τ+n}, n-step SARSA's return of the step at @param tau in @param episode (equation 7.4)
    value_type n_step_return(const markov_chain & episode,
                             std::size_t tau,
                             const policy<state_class,action_class,value_type> & policy_map) const;

    /// @return ρ_{τ+1:τ+n} for the step at @param tau in @param episode
    value_type importance_ratio(const markov_chain & episode,
                                std::size_t tau,
                                const policy<state_class,action_class,value_type> & policy_map) const;

    /**
     * @brief learn from an episode played by the behaviour policy - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link (see `n_step_sarsa`)
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);
};

/**
 * @struct n_step_tree_backup This is the n-step tree backup algorithm (Sutton & Barto, Section 7.5)
 * @brief n-step tree backup learns off-policy without importance sampling: at every step, it uses
 *        the values of the actions not taken, and follows the action taken, as π weighs them
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param n is the number of rewards each update looks ahead
 * @param probabilities is the policy π being learnt
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * After each reward, the return takes the values of every action π might have taken instead
 * of the one taken, each weighted by its probability, and carries on with the action taken,
 * weighted by its probability too (equation 7.16):
 *
 * G_{k:τ+n} = r_k + γ * Σ_{a ≠ a_{k+1}} π(a | s_{k+1}) Q(s_{k+1}, a) + γ * π(a_{k+1} | s_{k+1}) * G_{k+1:τ+n}
 *
 * which ends, after the n rewards, with the expected value of the state reached:
 * G_{τ+n-1:τ+n} = r_{τ+n-1} + γ * V̄(s_{τ+n}), or with the last reward r_{T-1} if the episode ends
 * sooner. Then Q(s_τ,a_τ) = Q(s_τ,a_τ) + α * (G_{τ:τ+n} - Q(s_τ,a_τ)).
 *
 * It never needs b: an action π would not take (π(a_{k+1} | s_{k+1}) = 0) simply cuts the return
 * there. So it learns π's values from episodes played by any policy, without the high variance of
 * importance sampling. With n = 1 this is `expected_sarsa`; with a greedy π, it is
 * Q-Learning's target followed for as long as the agent took the greedy actions.
 *
 * `probabilities` is as in `n_step_expected_sarsa`. Episodes cut short and learning online
 * work as for `n_step_sarsa`.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct n_step_tree_backup
{
    /// the actions the agent may take in a state, each with the probability of taking it
    using distribution = std::vector<std::pair<action_class,value_type>>;
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// n: the number of rewards each update looks ahead, at least 1 (1 is Expected SARSA)
    unsigned int n = 1;
    /**
     * optional: the learning rate of each update, given the number of times k = 1, 2, ...
     * the state/action pair has been updated, e.g. `relearn::polynomial_decay<>()` for α = 1/k.
     * If it is not set, `alpha` is used for every update.
     */
    std::function<value_type(std::size_t)> learning_rate;
    /// the number of updates of each state/action pair (counted only when `learning_rate` is set)
    visit_counter<state_class,action_class> visits;
    /// the policy π(a | s) being learnt - if not set, the greedy policy is used
    std::function<distribution(const state_class &)> probabilities;

    /**
     * @brief ctor with a learning rate @param initial_alpha and a discount rate @param initial_gamma,
     *        both 0.9 by default, the number of steps @param initial_n (1 by default),
     *        and the policy @param pi (`probabilities`), not set by default
     */
    n_step_tree_backup(value_type initial_alpha = 0.9,
                       value_type initial_gamma = 0.9,
                       unsigned int initial_n = 1,
                       std::function<distribution(const state_class &)> pi = nullptr);

    /// @return G_{τ:τ+n}, the tree backup return of the step at @param tau in @param episode
    value_type n_step_return(const markov_chain & episode,
                             std::size_t tau,
                             const policy<state_class,action_class,value_type> & policy_map) const;

    /**
     * @brief learn from an episode, played by π or by any other policy - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link (see `n_step_sarsa`)
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);
};

/**
 * @struct n_step_q_sigma This is n-step Q(σ) (Sutton & Barto, Section 7.6)
 * @brief n-step Q(σ) unifies the n-step methods: at every step it samples the action taken (σ = 1),
 *        as SARSA does, or takes the expectation over all actions (σ = 0), as tree backup does, or a mix
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param n is the number of rewards each update looks ahead
 * @param sigma is σ, how much to sample: from 0 (tree backup) to 1 (SARSA)
 * @param probabilities is the policy π being learnt
 * @param behaviour is the policy b that played the episodes, if it is not π (off-policy)
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * The return of Q(σ), after each reward (equation 7.17):
 *
 * G_{k:τ+n} = r_k + γ * V̄(s_{k+1}) + γ * (σ_{k+1} ρ_{k+1} + (1 - σ_{k+1}) π(a_{k+1} | s_{k+1})) * (G_{k+1:τ+n} - Q(s_{k+1}, a_{k+1}))
 *
 * ending with G_{τ+n:τ+n} = Q(s_{τ+n}, a_{τ+n}), or with the last reward r_{T-1} if the episode ends
 * sooner, and with ρ_k = π(a_k | s_k) / b(a_k | s_k) (1 on-policy). Then
 * Q(s_τ,a_τ) = Q(s_τ,a_τ) + α * (G_{τ:τ+n} - Q(s_τ,a_τ)).
 *
 * - σ = 0 everywhere is exactly `n_step_tree_backup`
 * - σ = 1 everywhere is n-step SARSA with per-decision control variates (equation 7.14): it samples
 *   every action taken, like n-step SARSA, and adds V̄(s) - Q(s, a) at each one, which is 0 on average,
 *   so on-policy it learns the same values as n-step SARSA; off-policy, it is importance
 *   sampling with each factor applied to its own step only
 * - with n = 1, whatever σ is, it is `expected_sarsa`: the last action of a return is always taken in expectation
 *
 * σ can change from step to step: `sigma_of(s, a)`, if it is set, gives σ for the step that took
 * a in s, and `sigma` otherwise. `probabilities` and `behaviour` are as in `n_step_expected_sarsa`.
 * Episodes cut short and learning online work as for `n_step_sarsa`.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct n_step_q_sigma
{
    /// the actions the agent may take in a state, each with the probability of taking it
    using distribution = std::vector<std::pair<action_class,value_type>>;
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// n: the number of rewards each update looks ahead, at least 1
    unsigned int n = 1;
    /// σ: how much each step samples the action taken, from 0 (tree backup) to 1 (SARSA)
    value_type sigma = 0.5;
    /// optional: σ for the step that took action a in state s, instead of `sigma`
    std::function<value_type(const state_class &, const action_class &)> sigma_of;
    /**
     * optional: the learning rate of each update, given the number of times k = 1, 2, ...
     * the state/action pair has been updated, e.g. `relearn::polynomial_decay<>()` for α = 1/k.
     * If it is not set, `alpha` is used for every update.
     */
    std::function<value_type(std::size_t)> learning_rate;
    /// the number of updates of each state/action pair (counted only when `learning_rate` is set)
    visit_counter<state_class,action_class> visits;
    /// the policy π(a | s) being learnt - if not set, the greedy policy is used
    std::function<distribution(const state_class &)> probabilities;
    /// optional: b(a | s), the probability that the policy which played the episodes takes a in s (off-policy)
    std::function<value_type(const state_class &, const action_class &)> behaviour;

    /**
     * @brief ctor with a learning rate @param initial_alpha and a discount rate @param initial_gamma,
     *        both 0.9 by default, the number of steps @param initial_n (1 by default), σ @param initial_sigma
     *        (0.5 by default), and the policy @param pi (`probabilities`), not set by default
     */
    n_step_q_sigma(value_type initial_alpha = 0.9,
                   value_type initial_gamma = 0.9,
                   unsigned int initial_n = 1,
                   value_type initial_sigma = 0.5,
                   std::function<distribution(const state_class &)> pi = nullptr);

    /// @return G_{τ:τ+n}, the Q(σ) return of the step at @param tau in @param episode
    value_type n_step_return(const markov_chain & episode,
                             std::size_t tau,
                             const policy<state_class,action_class,value_type> & policy_map) const;

    /**
     * @brief learn from an episode - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link (see `n_step_sarsa`)
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);
};

/********************************************************************************
 ***********               Template Implementations
 ********************************************************************************/

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
n_step_sarsa<state_class,action_class,markov_chain,value_type
            >::n_step_sarsa(value_type initial_alpha,
                            value_type initial_gamma,
                            unsigned int initial_n)
: alpha(initial_alpha),
  gamma(initial_gamma),
  n(initial_n)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type n_step_sarsa<state_class,action_class,markov_chain,value_type
                       >::n_step_return(const markov_chain & episode,
                                        std::size_t tau,
                                        const policy<state_class,action_class,value_type> & policy_map) const
{
    // r_τ + ... + γ^{n-1} * r_{τ+n-1}, or up to the end of the episode
    value_type discount = 1;
    value_type G = discounted_rewards(episode, tau, n, gamma, discount);
    // if the episode goes on after the n rewards: G = G + γ^n * Q(s_{τ+n}, a_{τ+n})
    if (tau + n < episode.size()) {
        const auto & later = episode[tau + n];
        G += discount * policy_map.value(later.state, later.action);
    }
    return G;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void n_step_sarsa<state_class,action_class,markov_chain,value_type
                 >::operator()(markov_chain episode,
                               policy<state_class,action_class,value_type> & policy_map,
                               bool terminated)
{
    assert(n >= 1 && "n_step_sarsa needs n >= 1");
    // an episode that ended: every step is learnt, τ = 0 ... T-1.
    // Otherwise the last link is not a step, and step τ is learnt only if its n rewards
    // are known, that is if τ + n is at most the index of the last link
    std::size_t learnt = episode.size();
    if (!terminated) {
        learnt = episode.size() > n ? episode.size() - n : 0;
    }
    for (std::size_t tau = 0; tau < learnt; tau++) {
        const auto & step = episode[tau];
        // α: `alpha`, or the scheduled learning rate of this update of (s_τ, a_τ)
        auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
        // Q(s_τ,a_τ) = Q(s_τ,a_τ) + α * (G_{τ:τ+n} - Q(s_τ,a_τ))
        auto G = n_step_return(episode, tau, policy_map);
        auto q = policy_map.value(step.state, step.action);
        policy_map.update(step.state, step.action, q + rate * (G - q));
    }
}

template <class state_class,
          class action_class,
          typename value_type>
std::vector<std::pair<action_class,value_type>>
    target_policy(const std::function<std::vector<std::pair<action_class,value_type>>(const state_class &)> & probabilities,
                  const state_class & s_t,
                  const policy<state_class,action_class,value_type> & policy_map)
{
    if (probabilities) {
        return probabilities(s_t);
    }
    // no policy given: the greedy one, on the actions stored for s_t
    std::vector<std::pair<action_class,value_type>> pi;
    const auto best = policy_map.best_value(s_t);
    if (std::isnan(best)) {
        return pi;
    }
    // the actions of highest value share the probability equally
    for (const auto & item : policy_map.actions(s_t)) {
        if (item.second == best) {
            pi.push_back(std::make_pair(item.first, value_type(1)));
        }
    }
    for (auto & item : pi) {
        item.second /= static_cast<value_type>(pi.size());
    }
    return pi;
}

template <class action_class,
          typename value_type>
value_type probability_of(const std::vector<std::pair<action_class,value_type>> & pi,
                          const action_class & a_t)
{
    for (const auto & item : pi) {
        if (item.first == a_t) {
            return item.second;
        }
    }
    return value_type(0);
}

template <class state_class,
          class action_class,
          typename value_type>
value_type expected_value_of(const std::vector<std::pair<action_class,value_type>> & pi,
                             const state_class & s_t,
                             const policy<state_class,action_class,value_type> & policy_map)
{
    // Σ_a π(a | s_t) Q(s_t, a): each action's value, weighted by its probability
    value_type expected = 0;
    for (const auto & item : pi) {
        expected += item.second * policy_map.value(s_t, item.first);
    }
    return expected;
}

template <typename markov_chain,
          typename value_type>
value_type discounted_rewards(const markov_chain & episode,
                              std::size_t tau,
                              unsigned int n,
                              value_type gamma,
                              value_type & discount)
{
    // the rewards r_τ ... r_{τ+n-1}, or up to the end of the episode if it is nearer
    const std::size_t stop = std::min<std::size_t>(tau + n, episode.size());
    value_type G = 0;
    discount = 1;
    for (std::size_t i = tau; i < stop; i++) {
        // G = G + γ^{i-τ} * r_i
        G += discount * episode[i].state.reward();
        discount *= gamma;
    }
    return G;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
n_step_expected_sarsa<state_class,action_class,markov_chain,value_type
                     >::n_step_expected_sarsa(value_type initial_alpha,
                                              value_type initial_gamma,
                                              unsigned int initial_n,
                                              std::function<distribution(const state_class &)> pi)
: alpha(initial_alpha),
  gamma(initial_gamma),
  n(initial_n),
  probabilities(pi)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type n_step_expected_sarsa<state_class,action_class,markov_chain,value_type
                                >::expected_value(const state_class & s_t,
                                                  const policy<state_class,action_class,value_type> & policy_map) const
{
    return expected_value_of(target_policy(probabilities, s_t, policy_map), s_t, policy_map);
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type n_step_expected_sarsa<state_class,action_class,markov_chain,value_type
                                >::n_step_return(const markov_chain & episode,
                                                 std::size_t tau,
                                                 const policy<state_class,action_class,value_type> & policy_map) const
{
    // r_τ + ... + γ^{n-1} * r_{τ+n-1}, or up to the end of the episode
    value_type discount = 1;
    value_type G = discounted_rewards(episode, tau, n, gamma, discount);
    // if the episode goes on after the n rewards: G = G + γ^n * V̄(s_{τ+n})
    if (tau + n < episode.size()) {
        G += discount * expected_value(episode[tau + n].state, policy_map);
    }
    return G;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type n_step_expected_sarsa<state_class,action_class,markov_chain,value_type
                                >::importance_ratio(const markov_chain & episode,
                                                    std::size_t tau,
                                                    const policy<state_class,action_class,value_type> & policy_map) const
{
    if (!behaviour) {
        return value_type(1);
    }
    // ρ_{τ+1:τ+n-1}: the actions a_{τ+1} ... a_{τ+n-1}, those of the episode's steps only
    const std::size_t last = std::min<std::size_t>(tau + n - 1, episode.size() - 1);
    value_type rho = 1;
    for (std::size_t k = tau + 1; k <= last; k++) {
        const auto & step = episode[k];
        const auto b = behaviour(step.state, step.action);
        assert(b > 0 && "the behaviour policy took an action it gives no probability to");
        rho *= probability_of(target_policy(probabilities, step.state, policy_map), step.action) / b;
    }
    return rho;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void n_step_expected_sarsa<state_class,action_class,markov_chain,value_type
                          >::operator()(markov_chain episode,
                                        policy<state_class,action_class,value_type> & policy_map,
                                        bool terminated)
{
    assert(n >= 1 && "n_step_expected_sarsa needs n >= 1");
    // the steps learnt: all of them if the episode ended, otherwise those whose n rewards are known
    std::size_t learnt = episode.size();
    if (!terminated) {
        learnt = episode.size() > n ? episode.size() - n : 0;
    }
    for (std::size_t tau = 0; tau < learnt; tau++) {
        const auto & step = episode[tau];
        // ρ: 1 on-policy; off-policy, a step π would not have played teaches nothing about π
        auto rho = importance_ratio(episode, tau, policy_map);
        if (rho == 0) {
            continue;
        }
        // α: `alpha`, or the scheduled learning rate of this update of (s_τ, a_τ)
        auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
        // Q(s_τ,a_τ) = Q(s_τ,a_τ) + α * ρ * (G_{τ:τ+n} - Q(s_τ,a_τ))
        auto G = n_step_return(episode, tau, policy_map);
        auto q = policy_map.value(step.state, step.action);
        policy_map.update(step.state, step.action, q + rate * rho * (G - q));
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
off_policy_n_step_sarsa<state_class,action_class,markov_chain,value_type
                       >::off_policy_n_step_sarsa(value_type initial_alpha,
                                                  value_type initial_gamma,
                                                  unsigned int initial_n,
                                                  std::function<distribution(const state_class &)> pi,
                                                  std::function<value_type(const state_class &, const action_class &)> b)
: alpha(initial_alpha),
  gamma(initial_gamma),
  n(initial_n),
  probabilities(pi),
  behaviour(b)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type off_policy_n_step_sarsa<state_class,action_class,markov_chain,value_type
                                  >::n_step_return(const markov_chain & episode,
                                                   std::size_t tau,
                                                   const policy<state_class,action_class,value_type> & policy_map) const
{
    // r_τ + ... + γ^{n-1} * r_{τ+n-1}, or up to the end of the episode
    value_type discount = 1;
    value_type G = discounted_rewards(episode, tau, n, gamma, discount);
    // if the episode goes on after the n rewards: G = G + γ^n * Q(s_{τ+n}, a_{τ+n})
    if (tau + n < episode.size()) {
        const auto & later = episode[tau + n];
        G += discount * policy_map.value(later.state, later.action);
    }
    return G;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type off_policy_n_step_sarsa<state_class,action_class,markov_chain,value_type
                                  >::importance_ratio(const markov_chain & episode,
                                                      std::size_t tau,
                                                      const policy<state_class,action_class,value_type> & policy_map) const
{
    // ρ_{τ+1:τ+n}: the actions a_{τ+1} ... a_{τ+n}, including the one whose value ends the return,
    // but not after the last step of an episode that ended
    const std::size_t last = std::min<std::size_t>(tau + n, episode.size() - 1);
    value_type rho = 1;
    for (std::size_t k = tau + 1; k <= last; k++) {
        const auto & step = episode[k];
        const auto b = behaviour(step.state, step.action);
        assert(b > 0 && "the behaviour policy took an action it gives no probability to");
        rho *= probability_of(target_policy(probabilities, step.state, policy_map), step.action) / b;
    }
    return rho;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void off_policy_n_step_sarsa<state_class,action_class,markov_chain,value_type
                            >::operator()(markov_chain episode,
                                          policy<state_class,action_class,value_type> & policy_map,
                                          bool terminated)
{
    assert(n >= 1 && "off_policy_n_step_sarsa needs n >= 1");
    assert(behaviour && "off_policy_n_step_sarsa needs the behaviour policy b(a | s)");
    // the steps learnt: all of them if the episode ended, otherwise those whose n rewards are known
    std::size_t learnt = episode.size();
    if (!terminated) {
        learnt = episode.size() > n ? episode.size() - n : 0;
    }
    for (std::size_t tau = 0; tau < learnt; tau++) {
        const auto & step = episode[tau];
        // ρ_{τ+1:τ+n}: a step π would not have played teaches nothing about π
        auto rho = importance_ratio(episode, tau, policy_map);
        if (rho == 0) {
            continue;
        }
        // α: `alpha`, or the scheduled learning rate of this update of (s_τ, a_τ)
        auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
        // Q(s_τ,a_τ) = Q(s_τ,a_τ) + α * ρ_{τ+1:τ+n} * (G_{τ:τ+n} - Q(s_τ,a_τ))
        auto G = n_step_return(episode, tau, policy_map);
        auto q = policy_map.value(step.state, step.action);
        policy_map.update(step.state, step.action, q + rate * rho * (G - q));
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
n_step_tree_backup<state_class,action_class,markov_chain,value_type
                  >::n_step_tree_backup(value_type initial_alpha,
                                        value_type initial_gamma,
                                        unsigned int initial_n,
                                        std::function<distribution(const state_class &)> pi)
: alpha(initial_alpha),
  gamma(initial_gamma),
  n(initial_n),
  probabilities(pi)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type n_step_tree_backup<state_class,action_class,markov_chain,value_type
                             >::n_step_return(const markov_chain & episode,
                                              std::size_t tau,
                                              const policy<state_class,action_class,value_type> & policy_map) const
{
    // the last reward of the return: r_{τ+n-1}, or r_{T-1} if the episode ends sooner
    const std::size_t end = std::min<std::size_t>(tau + n, episode.size());
    std::size_t k = end - 1;
    value_type G = episode[k].state.reward();
    if (end < episode.size()) {
        // the episode goes on: G_{τ+n-1:τ+n} = r_{τ+n-1} + γ * V̄(s_{τ+n})
        const auto & s_next = episode[end].state;
        G += gamma * expected_value_of(target_policy(probabilities, s_next, policy_map), s_next, policy_map);
    }
    // then back to τ, one reward at a time (equation 7.16)
    while (k > tau) {
        k--;
        const auto & next = episode[k + 1];
        auto pi = target_policy(probabilities, next.state, policy_map);
        // Σ_{a ≠ a_{k+1}} π(a | s_{k+1}) Q(s_{k+1}, a): the actions not taken
        value_type others = 0;
        for (const auto & item : pi) {
            if (!(item.first == next.action)) {
                others += item.second * policy_map.value(next.state, item.first);
            }
        }
        // G_{k:τ+n} = r_k + γ * others + γ * π(a_{k+1} | s_{k+1}) * G_{k+1:τ+n}
        G = episode[k].state.reward() + gamma * others + gamma * probability_of(pi, next.action) * G;
    }
    return G;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void n_step_tree_backup<state_class,action_class,markov_chain,value_type
                       >::operator()(markov_chain episode,
                                     policy<state_class,action_class,value_type> & policy_map,
                                     bool terminated)
{
    assert(n >= 1 && "n_step_tree_backup needs n >= 1");
    // the steps learnt: all of them if the episode ended, otherwise those whose n rewards are known
    std::size_t learnt = episode.size();
    if (!terminated) {
        learnt = episode.size() > n ? episode.size() - n : 0;
    }
    for (std::size_t tau = 0; tau < learnt; tau++) {
        const auto & step = episode[tau];
        // α: `alpha`, or the scheduled learning rate of this update of (s_τ, a_τ)
        auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
        // Q(s_τ,a_τ) = Q(s_τ,a_τ) + α * (G_{τ:τ+n} - Q(s_τ,a_τ))
        auto G = n_step_return(episode, tau, policy_map);
        auto q = policy_map.value(step.state, step.action);
        policy_map.update(step.state, step.action, q + rate * (G - q));
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
n_step_q_sigma<state_class,action_class,markov_chain,value_type
              >::n_step_q_sigma(value_type initial_alpha,
                                value_type initial_gamma,
                                unsigned int initial_n,
                                value_type initial_sigma,
                                std::function<distribution(const state_class &)> pi)
: alpha(initial_alpha),
  gamma(initial_gamma),
  n(initial_n),
  sigma(initial_sigma),
  probabilities(pi)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type n_step_q_sigma<state_class,action_class,markov_chain,value_type
                         >::n_step_return(const markov_chain & episode,
                                          std::size_t tau,
                                          const policy<state_class,action_class,value_type> & policy_map) const
{
    // where the return ends: after n rewards, at s_{τ+n}, or at the end of the episode if it is nearer
    const std::size_t end = std::min<std::size_t>(tau + n, episode.size());
    std::size_t k = end;
    value_type G = 0;
    if (end < episode.size()) {
        // the episode goes on: G_{τ+n:τ+n} = Q(s_{τ+n}, a_{τ+n})
        G = policy_map.value(episode[end].state, episode[end].action);
    }
    else {
        // the episode ends first: G_{T-1:T} = r_{T-1}
        k = end - 1;
        G = episode[k].state.reward();
    }
    // then back to τ, one reward at a time (equation 7.17)
    while (k > tau) {
        k--;
        const auto & next = episode[k + 1];
        auto pi = target_policy(probabilities, next.state, policy_map);
        const auto pi_taken = probability_of(pi, next.action);
        // σ_{k+1}, and ρ_{k+1} = π(a_{k+1} | s_{k+1}) / b(a_{k+1} | s_{k+1}), or 1 on-policy
        const auto s = sigma_of ? sigma_of(next.state, next.action) : sigma;
        assert(s >= 0 && s <= 1 && "n_step_q_sigma needs 0 <= sigma <= 1");
        value_type rho = 1;
        if (behaviour) {
            const auto b = behaviour(next.state, next.action);
            assert(b > 0 && "the behaviour policy took an action it gives no probability to");
            rho = pi_taken / b;
        }
        // G_{k:τ+n} = r_k + γ * V̄(s_{k+1}) + γ * (σ ρ + (1 - σ) π(a_{k+1} | s_{k+1})) * (G_{k+1:τ+n} - Q(s_{k+1}, a_{k+1}))
        const auto v_bar = expected_value_of(pi, next.state, policy_map);
        const auto q_taken = policy_map.value(next.state, next.action);
        G = episode[k].state.reward() + gamma * v_bar + gamma * (s * rho + (1 - s) * pi_taken) * (G - q_taken);
    }
    return G;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void n_step_q_sigma<state_class,action_class,markov_chain,value_type
                   >::operator()(markov_chain episode,
                                 policy<state_class,action_class,value_type> & policy_map,
                                 bool terminated)
{
    assert(n >= 1 && "n_step_q_sigma needs n >= 1");
    // the steps learnt: all of them if the episode ended, otherwise those whose n rewards are known
    std::size_t learnt = episode.size();
    if (!terminated) {
        learnt = episode.size() > n ? episode.size() - n : 0;
    }
    for (std::size_t tau = 0; tau < learnt; tau++) {
        const auto & step = episode[tau];
        // α: `alpha`, or the scheduled learning rate of this update of (s_τ, a_τ)
        auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
        // Q(s_τ,a_τ) = Q(s_τ,a_τ) + α * (G_{τ:τ+n} - Q(s_τ,a_τ))
        auto G = n_step_return(episode, tau, policy_map);
        auto q = policy_map.value(step.state, step.action);
        policy_map.update(step.state, step.action, q + rate * (G - q));
    }
}
} // end of namespace
#endif
