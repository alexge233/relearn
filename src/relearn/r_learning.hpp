#ifndef RELEARN_R_LEARNING_HPP
#define RELEARN_R_LEARNING_HPP
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
 * @file relearn/r_learning.hpp
 * @brief learning the average reward of tasks that never end: R-Learning (`r_learning`)
 *        and differential Sarsa (`differential_sarsa`)
 */
#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
#include <vector>

namespace relearn
{
/**
 * @struct r_learning This is R-Learning (Schwartz, 1993), as in Sutton & Barto, 1st edition, Section 6.7
 * @brief R-Learning learns, after every step of a task that never ends, the values of the actions
 *        and the average reward per step ρ
 * @param alpha is the learning rate of the values
 * @param beta is the learning rate of the average reward ρ
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * A task that never ends (a *continuing* task) has no final reward to work towards,
 * and without a discount its return is infinite. R-Learning instead learns ρ, the
 * reward the agent earns per step on average, and values each action by how much
 * more than ρ it earns from then on (its *relative*, or differential, value).
 * For each step (s_t, a_t, r_t, s_{t+1}), in this order (Figure 6.16 of the 1st edition):
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t - ρ + max_a Q(s_{t+1}, a) - Q(s_t,a_t))
 *
 * and then, only if a_t is (now) a greedy action of s_t, Q(s_t,a_t) = max_a Q(s_t,a):
 *
 * ρ = ρ + β * (r_t - ρ + max_a Q(s_{t+1}, a) - max_a Q(s_t, a))
 *
 * using the values just updated. Exploring actions do not move ρ, so ρ learns the average
 * reward of the greedy policy: R-Learning is **off-policy**, like Q-learning.
 *
 * ρ is part of the learner (`rho`), and changes with every greedy step it learns.
 * The maxima are taken over the actions stored for the state in `policy_map`, or, if
 * `legal_actions` is set, over every legal action, counting 0 for those not stored.
 *
 * A continuing task is learnt one step at a time, e.g. with `games::play_online`, which
 * gives the learner (s_t, a_t) and (s_{t+1}, a_{t+1}) after every step. So the last link of an
 * episode only serves as the next state of the link before it, unless `terminated` is true.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct r_learning
{
    /// α: the learning rate of the values - you may change this as you learn
    value_type alpha = 0.1;
    /// β: the learning rate of the average reward - you may change this as you learn
    value_type beta = 0.01;
    /// ρ: the average reward per step learnt so far
    value_type rho = 0;
    /// optional: every action that can be taken in a state, so that actions not tried yet count 0 in the maxima
    std::function<std::vector<action_class>(const state_class &)> legal_actions;

    /**
     * @brief ctor with the learning rates @param initial_alpha (α, 0.1 by default) and
     *        @param initial_beta (β, 0.01 by default), and the first average reward @param initial_rho (ρ, 0 by default)
     */
    r_learning(value_type initial_alpha = 0.1,
               value_type initial_beta = 0.01,
               value_type initial_rho = 0);

    /**
     * @return max_a Q(@param s_t, a), using the values of @param policy_map:
     *         over the `legal_actions` if set (0 for those not stored), otherwise over the
     *         actions stored for s_t (0 if there are none)
     */
    value_type best_value(const state_class & s_t,
                          const policy<state_class,action_class,value_type> & policy_map) const;

    /**
     * @brief learn the step at @param index of @param episode: update Q(s_t, a_t) in
     *        @param policy_map, and then ρ if a_t is greedy
     * @param ends is true if the episode ended after this step: the next state is then worth 0
     */
    void update(markov_chain & episode,
                unsigned int index,
                policy<state_class,action_class,value_type> & policy_map,
                bool ends);

    /**
     * @brief learn every step of @param episode, in order - @param policy_map and `rho` will be modified
     * @param terminated true if the episode really ended after its last link (a continuing task
     *        never does, so it is false by default): the last link is then learnt with a next
     *        state worth 0. Otherwise the last link is not learnt, and only serves as the next
     *        state of the link before it.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = false);
};

/**
 * @struct differential_sarsa This is differential Sarsa, tabular (Sutton & Barto, 2nd edition, Section 10.3)
 * @brief differential Sarsa learns, after every step of a task that never ends, the values of
 *        the actions taken and the average reward per step
 * @param alpha is the learning rate of the values
 * @param beta is the learning rate of the average reward
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * The 2nd edition replaces R-Learning by this on-policy method (it calls the average reward R̄,
 * here `rho` as in R-Learning). For each step (s_t, a_t, r_t, s_{t+1}, a_{t+1}):
 *
 * δ = r_t - ρ + Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)
 * ρ = ρ + β * δ
 * Q(s_t,a_t) = Q(s_t,a_t) + α * δ
 *
 * Unlike R-Learning, it uses the action a_{t+1} the agent took next, and ρ learns from
 * every step, exploration included: it learns the average reward of the policy the agent follows.
 * The book writes it with function approximation (semi-gradient); with one value per state and
 * action, the gradient is 1 for the pair updated and 0 for the others, which gives the above.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct differential_sarsa
{
    /// α: the learning rate of the values - you may change this as you learn
    value_type alpha = 0.01;
    /// β: the learning rate of the average reward - you may change this as you learn
    value_type beta = 0.01;
    /// ρ (R̄ in the 2nd edition): the average reward per step learnt so far
    value_type rho = 0;

    /**
     * @brief ctor with the learning rates @param initial_alpha (α) and @param initial_beta (β),
     *        both 0.01 by default, and the first average reward @param initial_rho (ρ, 0 by default)
     */
    differential_sarsa(value_type initial_alpha = 0.01,
                       value_type initial_beta = 0.01,
                       value_type initial_rho = 0);

    /**
     * @brief learn the step at @param index of @param episode: update ρ, and Q(s_t, a_t) in @param policy_map
     * @param ends is true if the episode ended after this step: the next state is then worth 0
     */
    void update(markov_chain & episode,
                unsigned int index,
                policy<state_class,action_class,value_type> & policy_map,
                bool ends);

    /**
     * @brief learn every step of @param episode, in order - @param policy_map and `rho` will be modified
     * @param terminated true if the episode really ended after its last link (a continuing task
     *        never does, so it is false by default): the last link is then learnt with a next
     *        state worth 0. Otherwise the last link is not learnt, and only serves as the next
     *        state and action of the link before it.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = false);
};

/********************************************************************************
 ***********               Template Implementations
 ********************************************************************************/

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
r_learning<state_class,action_class,markov_chain,value_type
          >::r_learning(value_type initial_alpha,
                        value_type initial_beta,
                        value_type initial_rho)
: alpha(initial_alpha),
  beta(initial_beta),
  rho(initial_rho)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type r_learning<state_class,action_class,markov_chain,value_type
                     >::best_value(const state_class & s_t,
                                   const policy<state_class,action_class,value_type> & policy_map) const
{
    // no legal actions given: the best of the actions stored for s_t
    if (!legal_actions) {
        auto best = policy_map.best_value(s_t);
        return std::isnan(best) ? value_type(0) : best;
    }
    // the best of the legal actions, those not stored counting 0
    auto legal = legal_actions(s_t);
    if (legal.empty()) {
        return value_type(0);
    }
    value_type best = policy_map.value(s_t, legal.front());
    for (const auto & a_t : legal) {
        best = std::max(best, policy_map.value(s_t, a_t));
    }
    return best;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void r_learning<state_class,action_class,markov_chain,value_type
               >::update(markov_chain & episode,
                         unsigned int index,
                         policy<state_class,action_class,value_type> & policy_map,
                         bool ends)
{
    const auto & step = episode[index];
    // r_t: the reward received after taking a_t in s_t
    const auto r = static_cast<value_type>(step.state.reward());
    // Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t - ρ + max_a Q(s_{t+1}, a) - Q(s_t,a_t))   (Figure 6.16)
    // a step that ends the episode has no next state: it is worth 0
    auto q      = policy_map.value(step.state, step.action);
    auto q_next = ends ? value_type(0) : best_value(episode[index + 1].state, policy_map);
    policy_map.update(step.state, step.action, q + alpha * (r - rho + q_next - q));
    // only a greedy action moves ρ: is a_t, with its new value, a best action of s_t?
    const auto q_best = best_value(step.state, policy_map);
    if (policy_map.value(step.state, step.action) < q_best) {
        return;
    }
    // ρ = ρ + β * (r_t - ρ + max_a Q(s_{t+1}, a) - max_a Q(s_t, a)), with the values just updated
    // (the next state's best value may have changed, if it is s_t itself)
    q_next = ends ? value_type(0) : best_value(episode[index + 1].state, policy_map);
    rho = rho + beta * (r - rho + q_next - q_best);
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void r_learning<state_class,action_class,markov_chain,value_type
               >::operator()(markov_chain episode,
                             policy<state_class,action_class,value_type> & policy_map,
                             bool terminated)
{
    // unless the episode ended, its last link has no next state yet: it is not learnt
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size()
                                                                   : episode.size() - 1;
    // learn the steps in order: each one changes the values and ρ the next one uses
    for (std::size_t i = 0; i < learnt; i++) {
        const bool ends = terminated && i == episode.size() - 1;
        update(episode, static_cast<unsigned int>(i), policy_map, ends);
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
differential_sarsa<state_class,action_class,markov_chain,value_type
                  >::differential_sarsa(value_type initial_alpha,
                                        value_type initial_beta,
                                        value_type initial_rho)
: alpha(initial_alpha),
  beta(initial_beta),
  rho(initial_rho)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void differential_sarsa<state_class,action_class,markov_chain,value_type
                       >::update(markov_chain & episode,
                                 unsigned int index,
                                 policy<state_class,action_class,value_type> & policy_map,
                                 bool ends)
{
    const auto & step = episode[index];
    // r_t: the reward received after taking a_t in s_t
    const auto r = static_cast<value_type>(step.state.reward());
    // Q(s_t, a_t), and Q(s_{t+1}, a_{t+1}) for the action the agent took next
    // (a step that ends the episode has no next state: it is worth 0)
    const auto q      = policy_map.value(step.state, step.action);
    const auto q_next = ends ? value_type(0)
                             : policy_map.value(episode[index + 1].state, episode[index + 1].action);
    // δ = r_t - ρ + Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)   (Section 10.3)
    const auto delta = r - rho + q_next - q;
    // ρ = ρ + β * δ, from every step
    rho = rho + beta * delta;
    // Q(s_t,a_t) = Q(s_t,a_t) + α * δ
    policy_map.update(step.state, step.action, q + alpha * delta);
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void differential_sarsa<state_class,action_class,markov_chain,value_type
                       >::operator()(markov_chain episode,
                                     policy<state_class,action_class,value_type> & policy_map,
                                     bool terminated)
{
    // unless the episode ended, its last link has no next state yet: it is not learnt
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size()
                                                                   : episode.size() - 1;
    // learn the steps in order: each one changes the values and ρ the next one uses
    for (std::size_t i = 0; i < learnt; i++) {
        const bool ends = terminated && i == episode.size() - 1;
        update(episode, static_cast<unsigned int>(i), policy_map, ends);
    }
}
} // end of namespace
#endif
