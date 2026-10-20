#ifndef RELEARN_Q_LEARNING_HPP
#define RELEARN_Q_LEARNING_HPP
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
 * @file relearn/q_learning.hpp
 * @brief Q-Learning (`q_learning`) and Q-Learning for non-deterministic systems (`q_probabilistic`)
 */
#include "core.hpp"
#include "schedules.hpp"
#include <cmath>
#include <deque>
#include <functional>
#include <tuple>
#include <unordered_map>

namespace relearn
{
/*******************************************************************************
 * @class q_learning This is the **deterministic** Q-Learning algorithm
 * @brief Q-Learning update algorithm sets policies using episodes (`markov_chain`)
 * @param gamma is the discount rate
 * @param alpha is the learning rate
 *
 * Episode convention: each link (s_t, a_t) carries in `s_t.reward()` the reward
 * r_t received *after* taking a_t in s_t (Sutton & Barto write it R_{t+1}).
 * The last link is the final decision of the episode.
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t + γ * max(Q(s_{t+1}, a)) - Q(s_t, a_t))
 *
 * and for the last link, which has no next state:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t - Q(s_t, a_t))
 *
 * In stochastic environments the Q-values converge to the expected discounted
 * return when, for every (s,a) pair, the learning rates used on its visits
 * satisfy Σ α = ∞ and Σ α² < ∞, e.g., α = 1/n with n the visits of that pair.
 * `alpha` is shared by all pairs, so a global schedule such as α = 1/k for
 * episode k only meets this if every pair keeps being visited regularly.
 * Set `learning_rate` instead to give each pair its own rate, from its own
 * number of updates: e.g. `polynomial_decay<>(1.0, 0.6)` for α = 1/n^0.6.
 */
template <class state_class, 
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double> 
struct q_learning
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
    q_learning(value_type initial_alpha = 0.9,
               value_type initial_gamma = 0.9);
    /// @brief the update rule of Q-learning
    triplet q_value(markov_chain &episode,
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

/**
 * @struct q_probabilistic This is the **non-deterministic** Q-Learning algorithm
 * @brief Q-Learning updates policy values using various episodes (`markov_chain`)
 * @param gamma is the discount rate
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * The algorithm remembers, for every observed (s_t, a_t), how many times it was
 * taken, the mean reward received, and how often each next state s_{t+1} followed.
 * From these it estimates the transition probability P(s_{t+1} | s_t, a_t) and
 * the expected reward E[r_t | s_t, a_t], and sets:
 *
 * Q(s_t,a_t) = E[r_t | s_t,a_t] + γ * Σ P(s_{t+1} | s_t,a_t) * max Q(s_{t+1}, a)
 *
 * where the sum runs over all observed next states (an episode ending after
 * (s_t, a_t) contributes 0). No learning rate is needed: the estimates converge
 * as more episodes are observed, provided every pair keeps being visited (each episode
 * only works Q out again for the pairs it visits).
 */
template <class state_class, 
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double> 
struct q_probabilistic
{
    // Q-triplet: state,action => value
    using triplet = std::tuple<state_class, 
                       		   action_class, 
                       		   value_type>;
    /// discount rate - you can change this as you process episodes
    value_type gamma = 0.9;
    /// default empty ctor - discount will be 0.9
    q_probabilistic() = default;
    /// ctor with a discount rate @param initial_gamma
    q_probabilistic(value_type initial_gamma);
    /// @brief the update rule of Q-learning
    triplet q_value(markov_chain &episode,
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
private:
    // map of state, frequency (s_t_+1)
    using frequency   = std::unordered_map<state_class,
                                           std::size_t,
                                           hasher<state_class>
                                          >;
    // what has been observed after taking a_t in s_t
    struct statistics
    {
        std::size_t visits = 0;     // times (s_t, a_t) was taken
        value_type reward_mean = 0; // running mean of the rewards r_t received
        frequency next;             // how often each s_{t+1} followed
    };
    // statistics of each action a_t taken in a state
    using transition  = std::unordered_map<action_class,
                                           statistics,
                                           hasher<action_class>
                                          >;
    // statistics of each observed (s_t, a_t)
    using observation = std::unordered_map<state_class,
                                           transition,
                                           hasher<state_class>
                                          >;
    observation memory_;
};

/********************************************************************************
 ***********               Template Implementations
 ********************************************************************************/

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
q_learning<state_class,action_class,markov_chain,value_type
          >::q_learning(value_type initial_alpha,
                        value_type initial_gamma)
: alpha(initial_alpha),
  gamma(initial_gamma)
{}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
typename q_learning<state_class,action_class,markov_chain,value_type>::triplet
            q_learning<state_class,action_class,markov_chain,value_type
					  >::q_value(markov_chain & episode, 
					   			 std::size_t index,
								 policy<state_class,action_class,value_type> & policy_map) 
{
    auto step = episode[index];
    // α: `alpha`, or the scheduled learning rate of this update of (s_t, a_t)
    auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
    if (index + 1 < episode.size())  {
        auto q      = policy_map.value(step.state, step.action);
        auto next   = episode[index + 1];
        auto q_next = policy_map.best_value(next.state);
        auto r      = step.state.reward();
        if (std::isnan(q_next)) q_next = 0.;
        return std::make_tuple(step.state, step.action, 
                               q + rate * (r + (gamma * q_next) - q));
    }
    else {
        // last decision of the episode: no next state to bootstrap from,
        // but the reward may be stochastic, so average it with the learning rate
        auto q = policy_map.value(step.state, step.action);
        auto r = step.state.reward();
        return std::make_tuple(step.state, step.action, q + rate * (r - q));
    }
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
void q_learning<state_class,action_class,markov_chain,value_type
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
q_probabilistic<state_class,action_class,markov_chain,value_type
               >::q_probabilistic(value_type initial_gamma)
: gamma(initial_gamma)
{}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
typename q_probabilistic<state_class,action_class,markov_chain,value_type>::triplet
            q_probabilistic<state_class,action_class,markov_chain,value_type
					       >::q_value(markov_chain & episode, 
                                      std::size_t index,
                                      policy<state_class,action_class,value_type> & policy_map) 
{
    auto step = episode[index];
    const auto & stats = memory_[step.state][step.action];
    if (stats.visits == 0) {
        return std::make_tuple(step.state, step.action, step.state.reward());
    }
    const auto visits = static_cast<value_type>(stats.visits);
    // expected reward E[r | s_t, a_t]
    value_type q = stats.reward_mean;
    // plus the discounted value of every observed next state, weighted by
    // P(s_{t+1} | s_t, a_t); episodes that ended after (s_t, a_t) contribute 0
    for (const auto & item : stats.next) {
        auto q_next = policy_map.best_value(item.first);
        if (std::isnan(q_next)) q_next = 0.;
        q += gamma * (static_cast<value_type>(item.second) / visits) * q_next;
    }
    return std::make_tuple(step.state, step.action, q);
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
void q_probabilistic<state_class,action_class,markov_chain,value_type
                    >::operator()(markov_chain episode, 
                 			      policy<state_class,action_class,value_type> & policy_map,
                                  bool terminated)
{
    // a cut-short episode does not learn its last link: it has no outcome yet
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size() 
                                                               : episode.size() - 1;
    // record what was observed: visits, rewards and next states of each (s_t, a_t)
    for (std::size_t i = 0; i < learnt; i++) {
        auto & stats = memory_[episode[i].state][episode[i].action];
        stats.visits += 1;
        // running mean rather than a sum: a float sum stops growing at 2^24 visits
        value_type r = episode[i].state.reward();
        stats.reward_mean += (r - stats.reward_mean) / static_cast<value_type>(stats.visits);
        if (i + 1 < episode.size()) {
            stats.next[episode[i + 1].state] += 1;
        }
    }
    for (std::size_t i = 0; i < learnt; i++) {
        auto update = q_value(episode, i, policy_map);
        policy_map.update(std::get<0>(update), 
                          std::get<1>(update), 
                          std::get<2>(update));
    }
}
} // end of namespace
#endif
