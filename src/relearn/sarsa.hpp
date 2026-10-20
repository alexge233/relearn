#ifndef RELEARN_SARSA_HPP
#define RELEARN_SARSA_HPP
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
 * @file relearn/sarsa.hpp
 * @brief SARSA (`sarsa`) and Expected SARSA (`expected_sarsa`)
 */
#include "core.hpp"
#include "schedules.hpp"
#include <cmath>
#include <deque>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

namespace relearn
{
/**
 * @struct sarsa This is the **on-policy** TD control algorithm (Sutton & Barto, Section 6.4)
 * @brief SARSA updates policy values using episodes (`markov_chain`)
 * @param gamma is the discount rate
 * @param alpha is the learning rate
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * SARSA learns from the quintuple (s_t, a_t, r_t, s_{t+1}, a_{t+1}), which gives
 * it its name. Where Q-learning uses the best action of the next state, SARSA
 * uses the action a_{t+1} that the agent actually took next:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t + γ * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))
 *
 * and for the last link, which has no next state:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t - Q(s_t, a_t))
 *
 * Because it learns the value of the actions really taken, exploration included,
 * SARSA learns the value of the policy the agent follows: it is **on-policy**.
 * So the episodes must be played with the policy being learnt (for example
 * ε-greedy on the values of `policy_map`). The extra last link of an episode cut
 * short must hold the action the agent would have taken next.
 */
template <class state_class, 
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double> 
struct sarsa
{
    using triplet = std::tuple<state_class, 
                               action_class, 
                               value_type>;
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /**
     * optional: the learning rate of each update, given the number of times n = 1, 2, ...
     * the state/action pair has been updated, e.g. `relearn::polynomial_decay<>()` for α = 1/n.
     * If it is not set, `alpha` is used for every update.
     */
    std::function<value_type(std::size_t)> learning_rate;
    /// the number of updates of each state/action pair (counted only when `learning_rate` is set)
    visit_counter<state_class,action_class> visits;
    /// ctor with a learning rate @param initial_alpha and a discount rate @param initial_gamma, both 0.9 by default
    sarsa(value_type initial_alpha = 0.9,
          value_type initial_gamma = 0.9);
    /// @brief the update rule of SARSA
    triplet q_value(markov_chain & episode,
                    std::size_t index,
                    policy<state_class,action_class,value_type> & policy_map);

    /**
     * @brief do the updating for an episode - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short (e.g., by a step limit): its last
     *        link is then not learnt as the end of the game, and only serves as
     *        the next state and action of the link before it.
     */
    void operator()(markov_chain episode, 
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);
};

/**
 * @struct expected_sarsa This is the Expected SARSA algorithm (Sutton & Barto, Section 6.6)
 * @brief Expected SARSA updates policy values using episodes (`markov_chain`)
 * @param gamma is the discount rate
 * @param alpha is the learning rate
 * @param probabilities is the policy π the agent follows
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * Instead of the action taken next (SARSA) or the best one (Q-learning),
 * Expected SARSA uses the value the next state is *expected* to have under the
 * policy π the agent follows: the average of its action values, each weighted
 * by the probability that π takes it:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t + γ * Σ_a π(a | s_{t+1}) Q(s_{t+1}, a) - Q(s_t, a_t))
 *
 * and for the last link, which has no next state:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t - Q(s_t, a_t))
 *
 * This removes the randomness of choosing a_{t+1}, so it learns more steadily
 * than SARSA, and it can use larger learning rates.
 *
 * relearn does not know which actions are legal in your problem, or how your
 * agent explores, so you give it π: `probabilities(s)` returns each action the
 * agent may take in state s, with the probability that it takes it.
 * If `probabilities` is not set, the greedy policy is used, and the update is
 * the same as Q-learning's.
 */
template <class state_class, 
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double> 
struct expected_sarsa
{
    using triplet = std::tuple<state_class, 
                               action_class, 
                               value_type>;
    /// the actions the agent may take in a state, each with the probability of taking it
    using distribution = std::vector<std::pair<action_class,value_type>>;
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /**
     * optional: the learning rate of each update, given the number of times n = 1, 2, ...
     * the state/action pair has been updated, e.g. `relearn::polynomial_decay<>()` for α = 1/n.
     * If it is not set, `alpha` is used for every update.
     */
    std::function<value_type(std::size_t)> learning_rate;
    /// the number of updates of each state/action pair (counted only when `learning_rate` is set)
    visit_counter<state_class,action_class> visits;
    /// the policy π(a | s) the agent follows - if not set, the greedy policy is used
    std::function<distribution(const state_class &)> probabilities;
    /**
     * @brief ctor with a learning rate @param initial_alpha, a discount rate @param initial_gamma,
     *        both 0.9 by default, and the policy @param pi (`probabilities`), not set by default
     */
    expected_sarsa(value_type initial_alpha = 0.9,
                   value_type initial_gamma = 0.9,
                   std::function<distribution(const state_class &)> pi = nullptr);

    /**
     * @return the expected value of @param s_t under the policy π:
     *         Σ_a π(a | s_t) Q(s_t, a), using the values of @param policy_map
     */
    value_type expected_value(const state_class & s_t,
                              const policy<state_class,action_class,value_type> & policy_map) const;
    /// @brief the update rule of Expected SARSA
    triplet q_value(markov_chain & episode,
                    std::size_t index,
                    policy<state_class,action_class,value_type> & policy_map);

    /**
     * @brief do the updating for an episode - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short (e.g., by a step limit): its last
     *        link is then not learnt as the end of the game, and only serves as
     *        the next state of the link before it.
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
sarsa<state_class,action_class,markov_chain,value_type
     >::sarsa(value_type initial_alpha,
              value_type initial_gamma)
: alpha(initial_alpha),
  gamma(initial_gamma)
{}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
expected_sarsa<state_class,action_class,markov_chain,value_type
              >::expected_sarsa(value_type initial_alpha,
                                value_type initial_gamma,
                                std::function<distribution(const state_class &)> pi)
: alpha(initial_alpha),
  gamma(initial_gamma),
  probabilities(pi)
{}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
typename sarsa<state_class,action_class,markov_chain,value_type>::triplet
            sarsa<state_class,action_class,markov_chain,value_type
                 >::q_value(markov_chain & episode, 
                            std::size_t index,
                            policy<state_class,action_class,value_type> & policy_map) 
{
    auto step = episode[index];
    // α: `alpha`, or the scheduled learning rate of this update of (s_t, a_t)
    auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
    // Q(s_t, a_t): the value we update
    auto q = policy_map.value(step.state, step.action);
    // r_t: the reward received after taking a_t in s_t
    auto r = step.state.reward();
    if (index + 1 < episode.size()) {
        // the next link holds s_{t+1}, and a_{t+1}: the action the agent took next
        auto next   = episode[index + 1];
        auto q_next = policy_map.value(next.state, next.action);
        return std::make_tuple(step.state, step.action, 
                               q + rate * (r + (gamma * q_next) - q));
    }
    else {
        // last decision of the episode: no next state to bootstrap from,
        // but the reward may be stochastic, so average it with the learning rate
        return std::make_tuple(step.state, step.action, q + rate * (r - q));
    }
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
void sarsa<state_class,action_class,markov_chain,value_type
          >::operator()(markov_chain episode, 
                        policy<state_class,action_class,value_type> & policy_map,
                        bool terminated)
{
    // a cut-short episode does not learn its last link: it has no outcome yet
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size() 
                                                               : episode.size() - 1;
    for (std::size_t i = 0; i < learnt; i++) {
        auto update = q_value(episode, i, policy_map);
        policy_map.update(std::get<0>(update), 
                          std::get<1>(update), 
                          std::get<2>(update));
    }
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
value_type expected_sarsa<state_class,action_class,markov_chain,value_type
                         >::expected_value(const state_class & s_t,
                                           const policy<state_class,action_class,value_type> & policy_map) const
{
    if (!probabilities) {
        // no policy given: follow the greedy one, and take the best value known
        auto best = policy_map.best_value(s_t);
        return std::isnan(best) ? value_type(0) : best;
    }
    // Σ_a π(a | s_t) Q(s_t, a): each action's value, weighted by its probability
    value_type expected = 0;
    for (const auto & item : probabilities(s_t)) {
        expected += item.second * policy_map.value(s_t, item.first);
    }
    return expected;
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
typename expected_sarsa<state_class,action_class,markov_chain,value_type>::triplet
            expected_sarsa<state_class,action_class,markov_chain,value_type
                          >::q_value(markov_chain & episode, 
                                     std::size_t index,
                                     policy<state_class,action_class,value_type> & policy_map) 
{
    auto step = episode[index];
    // α: `alpha`, or the scheduled learning rate of this update of (s_t, a_t)
    auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
    // Q(s_t, a_t): the value we update
    auto q = policy_map.value(step.state, step.action);
    // r_t: the reward received after taking a_t in s_t
    auto r = step.state.reward();
    if (index + 1 < episode.size()) {
        // the expected value of s_{t+1} under the policy the agent follows
        auto next   = episode[index + 1];
        auto v_next = expected_value(next.state, policy_map);
        return std::make_tuple(step.state, step.action, 
                               q + rate * (r + (gamma * v_next) - q));
    }
    else {
        // last decision of the episode: no next state to bootstrap from,
        // but the reward may be stochastic, so average it with the learning rate
        return std::make_tuple(step.state, step.action, q + rate * (r - q));
    }
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
void expected_sarsa<state_class,action_class,markov_chain,value_type
                   >::operator()(markov_chain episode, 
                                 policy<state_class,action_class,value_type> & policy_map,
                                 bool terminated)
{
    // a cut-short episode does not learn its last link: it has no outcome yet
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size() 
                                                               : episode.size() - 1;
    for (std::size_t i = 0; i < learnt; i++) {
        auto update = q_value(episode, i, policy_map);
        policy_map.update(std::get<0>(update), 
                          std::get<1>(update), 
                          std::get<2>(update));
    }
}
} // end of namespace
#endif
