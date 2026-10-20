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
 * @brief n-step SARSA (`n_step_sarsa`), and n-step TD prediction (`n_step_td`), which is TD(0) when n = 1
 */
#include "core.hpp"
#include "schedules.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <functional>

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
    // the rewards r_τ ... r_{τ+n-1}, or up to the end of the episode if it is nearer
    const std::size_t stop = std::min<std::size_t>(tau + n, episode.size());
    value_type G = 0;
    value_type discount = 1;
    for (std::size_t i = tau; i < stop; i++) {
        // G = G + γ^{i-τ} * r_i
        G += discount * episode[i].state.reward();
        discount *= gamma;
    }
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
} // end of namespace
#endif
