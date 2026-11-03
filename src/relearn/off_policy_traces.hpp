#ifndef RELEARN_OFF_POLICY_TRACES_HPP
#define RELEARN_OFF_POLICY_TRACES_HPP
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
 * @file relearn/off_policy_traces.hpp
 * @brief off-policy eligibility traces: off-policy TD(λ) with control variates (`off_policy_td_lambda`),
 *        off-policy Expected Sarsa(λ) with control variates (`off_policy_expected_sarsa_lambda`),
 *        and Tree-Backup(λ) (`tree_backup_lambda`)
 */
#include "core.hpp"
#include "eligibility_traces.hpp"
#include "n_step.hpp"
#include <cassert>
#include <cstddef>
#include <deque>
#include <functional>
#include <utility>
#include <vector>

namespace relearn
{
/**
 * @struct off_policy_td_lambda This is off-policy TD(λ) with per-decision importance sampling and
 *         control variates (Sutton & Barto, Section 12.9)
 * @brief TD(λ) that learns the state values v_π of a target policy π from episodes played by another, b
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param lambda is λ
 * @param target is π(a | s), the policy whose values are learnt
 * @param behaviour is b(a | s), the policy that played the episodes
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * With ρ_t = π(a_t | s_t) / b(a_t | s_t), the ratio of step t, the λ-return of Section 12.9 weights what
 * follows a state by its ratio, and keeps its value for the rest (the *control variate*):
 *
 * G^λs_t = ρ_t * (r_t + γ_{t+1} * ((1 - λ_{t+1}) * V(s_{t+1}) + λ_{t+1} * G^λs_{t+1})) + (1 - ρ_t) * V(s_t)
 *
 * so a step π would never take (ρ_t = 0) leaves V(s_t) where it is, instead of pulling it to 0 as plain
 * importance sampling would. Its backward view, at every step t, in order:
 *
 * δ = r_t + γ_{t+1} * V(s_{t+1}) - V(s_t)          (δ = r_t - V(s_t) at the end)
 * z(s) = ρ_t * (γ_t λ_t * z(s) + [s = s_t]) for every state
 * V(s) = V(s) + α * δ * z(s) for every state
 *
 * with every trace z starting at 0, and γ_t = γ(s_t), λ_t = λ(s_t, a_t). Offline (`offline` true, every δ
 * worked out with the values from before the episode), it makes exactly the changes of that λ-return;
 * online, nearly. With π = b (every ρ is 1) this is TD(λ), and with λ = 0, off-policy TD(0) with the
 * ratio of each step.
 *
 * The values are kept in the learner, since an off-policy prediction task has several actions per state:
 * `value(s)`. b is read for every link of an episode once, before anything is learnt, as the episode
 * was played with it. λ and γ may vary from state to state (Section 12.8): set `lambda_of` and `gamma_of`.
 * Each call learns one whole episode, as `sarsa_lambda` does; an episode cut short (`terminated` false)
 * ends with the state reached, whose value the last step bootstraps from.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct off_policy_td_lambda
{
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// λ, 0 ... 1
    value_type lambda = 0.9;
    /// learn offline: the values change only at the end of the episode
    bool offline = false;
    /// π(a | s): the probability that the policy whose values are learnt takes a in s - required
    std::function<value_type(const state_class &, const action_class &)> target;
    /// b(a | s): the probability that the policy which played the episodes takes a in s - required
    std::function<value_type(const state_class &, const action_class &)> behaviour;
    /// optional: λ(s, a), a λ for every state/action pair (Section 12.8), instead of `lambda`
    std::function<value_type(const state_class &, const action_class &)> lambda_of;
    /// optional: γ(s), a discount for every state (Section 12.8), instead of `gamma`
    std::function<value_type(const state_class &)> gamma_of;
    /// the traces z(s) of the last episode learnt, each of them kept as the trace of (s, action 0)
    eligibility_traces<state_class,action<int>,value_type> traces;

    /**
     * @brief ctor with a learning rate @param initial_alpha, a discount rate @param initial_gamma
     *        and @param initial_lambda, all 0.9 by default, the target policy @param pi and the behaviour policy @param b
     */
    off_policy_td_lambda(value_type initial_alpha = 0.9,
                         value_type initial_gamma = 0.9,
                         value_type initial_lambda = 0.9,
                         std::function<value_type(const state_class &, const action_class &)> pi = nullptr,
                         std::function<value_type(const state_class &, const action_class &)> b = nullptr);

    /**
     * @brief learn from a whole episode played by b
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short (e.g., by a step limit): its last
     *        link is then not learnt, and only gives the state the step before it reached.
     */
    void operator()(markov_chain episode,
                    bool terminated = true);

    /// @return V(@param s), 0 for a state never learnt
    value_type value(const state_class & s) const;

private:
    // V(s), kept as the value of (s, action 0)
    policy<state_class,action<int>,value_type> values_;
};

/**
 * @struct off_policy_expected_sarsa_lambda This is the action-value off-policy TD(λ) of Section 12.9
 *         (Sutton & Barto), with per-decision importance sampling and control variates
 * @brief Expected Sarsa(λ): the action values q_π of a target policy π, learnt from episodes played by another, b
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param lambda is λ
 * @param probabilities is the policy π being learnt (greedy if it is not set)
 * @param behaviour is b(a | s), the policy that played the episodes
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * With V̄(s) = Σ_a π(a | s) Q(s, a) and ρ_t = π(a_t | s_t) / b(a_t | s_t), the λ-return of Section 12.9 is
 *
 * G^λa_t = r_t + γ_{t+1} * ((1 - λ_{t+1}) * V̄(s_{t+1}) + λ_{t+1} * (ρ_{t+1} * G^λa_{t+1} + V̄(s_{t+1}) - ρ_{t+1} * Q(s_{t+1}, a_{t+1})))
 *
 * and its backward view, at every step t, in order:
 *
 * δ = r_t + γ_{t+1} * V̄(s_{t+1}) - Q(s_t, a_t)          (δ = r_t - Q(s_t, a_t) at the end)
 * z(s,a) = γ_t λ_t ρ_t * z(s,a) for every pair, then z(s_t,a_t) = z(s_t,a_t) + 1
 * Q(s,a) = Q(s,a) + α * δ * z(s,a) for every pair
 *
 * with every trace z starting at 0, and γ_t = γ(s_t), λ_t = λ(s_t, a_t). Offline (`offline` true) it makes
 * exactly the changes of that λ-return; online, nearly. With λ = 0 this is `expected_sarsa`; with λ = 1 and
 * offline it is n-step Q(σ) with σ = 1 (`n_step_q_sigma`) to the end of the episode, on an episode that never comes
 * back to a state (the n-step learners update as they go). A step π would not take
 * (ρ = 0) cuts the traces, as Watkins's Q(λ) does after a move that is not greedy; a step π takes more often
 * than b makes them grow, by ρ > 1, which is where the variance of importance sampling comes from.
 *
 * `probabilities` is as in `expected_sarsa`: each action π may take in s, with its probability; if it is not
 * set, π is greedy on the values, and V̄ is then the highest value of s; an action whose value is not stored yet
 * has π = 0 (Watkins's Q(λ) counts such a move as greedy). π is read with the values as they
 * are at each step (from before the episode, offline), and b for every link once, before anything is learnt.
 * λ and γ may vary from state to state (Section 12.8): set `lambda_of` and `gamma_of`. Each call learns one
 * whole episode, as `sarsa_lambda` does; an episode cut short (`terminated` false) ends with the state reached
 * and the action chosen next, which is not a step.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct off_policy_expected_sarsa_lambda
{
    /// the actions the agent may take in a state, each with the probability of taking it
    using distribution = std::vector<std::pair<action_class,value_type>>;
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// λ, 0 ... 1 (0 is Expected SARSA)
    value_type lambda = 0.9;
    /// learn offline: the values change only at the end of the episode
    bool offline = false;
    /// the policy π(a | s) being learnt - if not set, the greedy policy is used
    std::function<distribution(const state_class &)> probabilities;
    /// b(a | s): the probability that the policy which played the episodes takes a in s - required
    std::function<value_type(const state_class &, const action_class &)> behaviour;
    /// optional: λ(s, a), a λ for every state/action pair (Section 12.8), instead of `lambda`
    std::function<value_type(const state_class &, const action_class &)> lambda_of;
    /// optional: γ(s), a discount for every state (Section 12.8), instead of `gamma`
    std::function<value_type(const state_class &)> gamma_of;
    /// the traces z(s, a) of the last episode learnt
    eligibility_traces<state_class,action_class,value_type> traces;

    /**
     * @brief ctor with a learning rate @param initial_alpha, a discount rate @param initial_gamma
     *        and @param initial_lambda, all 0.9 by default, the policy being learnt @param pi
     *        (greedy if not set) and the behaviour policy @param b
     */
    off_policy_expected_sarsa_lambda(value_type initial_alpha = 0.9,
                                     value_type initial_gamma = 0.9,
                                     value_type initial_lambda = 0.9,
                                     std::function<distribution(const state_class &)> pi = nullptr,
                                     std::function<value_type(const state_class &, const action_class &)> b = nullptr);

    /**
     * @brief learn from a whole episode played by b - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short (e.g., by a step limit): its last
     *        link is then not learnt, and only gives the state the step before it reached.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);
};

/**
 * @struct tree_backup_lambda This is Tree-Backup(λ), TB(λ) (Sutton & Barto, Section 12.10)
 * @brief the action values q_π of a target policy π, learnt from episodes played by any policy,
 *        without importance sampling
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param lambda is λ
 * @param probabilities is the policy π being learnt (greedy if it is not set)
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * The λ-return of n-step tree backup (Section 7.5), averaged over n: after each reward, the values of the
 * actions π might have taken instead, and the return of the action taken, each weighted by its probability:
 *
 * G^λa_t = r_t + γ_{t+1} * ((1 - λ_{t+1}) * V̄(s_{t+1}) + λ_{t+1} * (Σ_{a ≠ a_{t+1}} π(a | s_{t+1}) Q(s_{t+1}, a) + π(a_{t+1} | s_{t+1}) * G^λa_{t+1}))
 *
 * Its backward view is `off_policy_expected_sarsa_lambda`'s, with π(a_t | s_t) in place of ρ_t:
 *
 * δ = r_t + γ_{t+1} * V̄(s_{t+1}) - Q(s_t, a_t)          (δ = r_t - Q(s_t, a_t) at the end)
 * z(s,a) = γ_t λ_t π(a_t | s_t) * z(s,a) for every pair, then z(s_t,a_t) = z(s_t,a_t) + 1
 * Q(s,a) = Q(s,a) + α * δ * z(s,a) for every pair
 *
 * It never needs b: the traces only ever shrink, by π, so it learns π's values from any behaviour without
 * the variance of importance sampling, but the traces fade fast whenever π is not close to deterministic,
 * even on-policy. With a greedy π, its traces are cut after a move that is not greedy, much as Watkins's Q(λ)'s
 * are (`watkins_q_lambda`), and after a move whose value is not stored yet. Offline (`offline` true) it makes exactly the changes of that λ-return; with
 * λ = 0 it is `expected_sarsa`, and with λ = 1 and offline, `n_step_tree_backup` to the end of the episode, on an
 * episode that never comes back to a state.
 *
 * `probabilities`, `lambda_of`, `gamma_of`, episodes cut short and each call learning a whole episode are as in
 * `off_policy_expected_sarsa_lambda`.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct tree_backup_lambda
{
    /// the actions the agent may take in a state, each with the probability of taking it
    using distribution = std::vector<std::pair<action_class,value_type>>;
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// λ, 0 ... 1 (0 is Expected SARSA)
    value_type lambda = 0.9;
    /// learn offline: the values change only at the end of the episode
    bool offline = false;
    /// the policy π(a | s) being learnt - if not set, the greedy policy is used
    std::function<distribution(const state_class &)> probabilities;
    /// optional: λ(s, a), a λ for every state/action pair (Section 12.8), instead of `lambda`
    std::function<value_type(const state_class &, const action_class &)> lambda_of;
    /// optional: γ(s), a discount for every state (Section 12.8), instead of `gamma`
    std::function<value_type(const state_class &)> gamma_of;
    /// the traces z(s, a) of the last episode learnt
    eligibility_traces<state_class,action_class,value_type> traces;

    /**
     * @brief ctor with a learning rate @param initial_alpha, a discount rate @param initial_gamma
     *        and @param initial_lambda, all 0.9 by default, and the policy being learnt @param pi (greedy if not set)
     */
    tree_backup_lambda(value_type initial_alpha = 0.9,
                       value_type initial_gamma = 0.9,
                       value_type initial_lambda = 0.9,
                       std::function<distribution(const state_class &)> pi = nullptr);

    /**
     * @brief learn from a whole episode, played by any policy - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short (e.g., by a step limit): its last
     *        link is then not learnt, and only gives the state the step before it reached.
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
off_policy_td_lambda<state_class,action_class,markov_chain,value_type
                    >::off_policy_td_lambda(value_type initial_alpha,
                                            value_type initial_gamma,
                                            value_type initial_lambda,
                                            std::function<value_type(const state_class &, const action_class &)> pi,
                                            std::function<value_type(const state_class &, const action_class &)> b)
: alpha(initial_alpha),
  gamma(initial_gamma),
  lambda(initial_lambda),
  target(pi),
  behaviour(b)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void off_policy_td_lambda<state_class,action_class,markov_chain,value_type
                         >::operator()(markov_chain episode,
                                       bool terminated)
{
    assert(target && behaviour && "off_policy_td_lambda needs π and b");
    // a new episode: every trace starts at 0
    traces.clear();
    // a cut-short episode does not learn its last link: it is not a step
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size()
                                                                   : episode.size() - 1;
    // b(a_t | s_t), read for every step before anything is learnt
    std::vector<value_type> b(learnt);
    for (std::size_t t = 0; t < learnt; t++) {
        b[t] = behaviour(episode[t].state, episode[t].action);
        assert(b[t] > 0 && "b must be able to take every action it took");
    }
    // offline, every δ is worked out with the values from before the episode
    const policy<state_class,action<int>,value_type> before = offline ? values_
                                                                      : policy<state_class,action<int>,value_type>();
    const auto & values = offline ? before : values_;
    const action<int> only(0);
    for (std::size_t t = 0; t < learnt; t++) {
        const auto & step = episode[t];
        // δ = r_t + γ_{t+1} V(s_{t+1}) - V(s_t), with nothing after the last step of an episode
        value_type delta = step.state.reward() - values.value(step.state, only);
        if (t + 1 < episode.size()) {
            const auto & next = episode[t + 1];
            delta += gamma_at(gamma_of, gamma, next.state) * values.value(next.state, only);
        }
        // z = ρ_t (γ_t λ_t z + [s = s_t])
        const value_type rho = target(step.state, step.action) / b[t];
        traces.decay(gamma_at(gamma_of, gamma, step.state) * lambda_at(lambda_of, lambda, step.state, step.action));
        traces.visit(step.state, only, false);
        traces.decay(rho);
        // V(s) = V(s) + α δ z(s), for every state
        traces.update(values_, alpha * delta);
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type off_policy_td_lambda<state_class,action_class,markov_chain,value_type
                               >::value(const state_class & s) const
{
    return values_.value(s, action<int>(0));
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
off_policy_expected_sarsa_lambda<state_class,action_class,markov_chain,value_type
                                >::off_policy_expected_sarsa_lambda(value_type initial_alpha,
                                                                    value_type initial_gamma,
                                                                    value_type initial_lambda,
                                                                    std::function<distribution(const state_class &)> pi,
                                                                    std::function<value_type(const state_class &, const action_class &)> b)
: alpha(initial_alpha),
  gamma(initial_gamma),
  lambda(initial_lambda),
  probabilities(pi),
  behaviour(b)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void off_policy_expected_sarsa_lambda<state_class,action_class,markov_chain,value_type
                                     >::operator()(markov_chain episode,
                                                   policy<state_class,action_class,value_type> & policy_map,
                                                   bool terminated)
{
    assert(behaviour && "off_policy_expected_sarsa_lambda needs b");
    // a new episode: every trace starts at 0
    traces.clear();
    // a cut-short episode does not learn its last link: it is not a step
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size()
                                                                   : episode.size() - 1;
    // b(a_t | s_t), read for every step before anything is learnt
    std::vector<value_type> b(learnt);
    for (std::size_t t = 0; t < learnt; t++) {
        b[t] = behaviour(episode[t].state, episode[t].action);
        assert(b[t] > 0 && "b must be able to take every action it took");
    }
    // offline, every δ (and π) is worked out with the values from before the episode
    const policy<state_class,action_class,value_type> before = offline ? policy_map
                                                                       : policy<state_class,action_class,value_type>();
    const auto & values = offline ? before : policy_map;
    for (std::size_t t = 0; t < learnt; t++) {
        const auto & step = episode[t];
        // δ = r_t + γ_{t+1} V̄(s_{t+1}) - Q(s_t, a_t), with nothing after the last step of an episode
        value_type delta = step.state.reward() - values.value(step.state, step.action);
        if (t + 1 < episode.size()) {
            const auto & next = episode[t + 1];
            const auto pi_next = target_policy(probabilities, next.state, values);
            delta += gamma_at(gamma_of, gamma, next.state) * expected_value_of(pi_next, next.state, values);
        }
        // z = γ_t λ_t ρ_t z, then z(s_t, a_t) = z(s_t, a_t) + 1
        const auto pi = target_policy(probabilities, step.state, values);
        const value_type rho = probability_of(pi, step.action) / b[t];
        traces.decay(gamma_at(gamma_of, gamma, step.state) * lambda_at(lambda_of, lambda, step.state, step.action) * rho);
        traces.visit(step.state, step.action, false);
        // Q(s,a) = Q(s,a) + α δ z(s,a), for every pair
        traces.update(policy_map, alpha * delta);
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
tree_backup_lambda<state_class,action_class,markov_chain,value_type
                  >::tree_backup_lambda(value_type initial_alpha,
                                        value_type initial_gamma,
                                        value_type initial_lambda,
                                        std::function<distribution(const state_class &)> pi)
: alpha(initial_alpha),
  gamma(initial_gamma),
  lambda(initial_lambda),
  probabilities(pi)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void tree_backup_lambda<state_class,action_class,markov_chain,value_type
                       >::operator()(markov_chain episode,
                                     policy<state_class,action_class,value_type> & policy_map,
                                     bool terminated)
{
    // a new episode: every trace starts at 0
    traces.clear();
    // a cut-short episode does not learn its last link: it is not a step
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size()
                                                                   : episode.size() - 1;
    // offline, every δ (and π) is worked out with the values from before the episode
    const policy<state_class,action_class,value_type> before = offline ? policy_map
                                                                       : policy<state_class,action_class,value_type>();
    const auto & values = offline ? before : policy_map;
    for (std::size_t t = 0; t < learnt; t++) {
        const auto & step = episode[t];
        // δ = r_t + γ_{t+1} V̄(s_{t+1}) - Q(s_t, a_t), with nothing after the last step of an episode
        value_type delta = step.state.reward() - values.value(step.state, step.action);
        if (t + 1 < episode.size()) {
            const auto & next = episode[t + 1];
            const auto pi_next = target_policy(probabilities, next.state, values);
            delta += gamma_at(gamma_of, gamma, next.state) * expected_value_of(pi_next, next.state, values);
        }
        // z = γ_t λ_t π(a_t | s_t) z, then z(s_t, a_t) = z(s_t, a_t) + 1
        const auto pi = target_policy(probabilities, step.state, values);
        traces.decay(gamma_at(gamma_of, gamma, step.state) * lambda_at(lambda_of, lambda, step.state, step.action)
                     * probability_of(pi, step.action));
        traces.visit(step.state, step.action, false);
        // Q(s,a) = Q(s,a) + α δ z(s,a), for every pair
        traces.update(policy_map, alpha * delta);
    }
}
} // end of namespace
#endif
