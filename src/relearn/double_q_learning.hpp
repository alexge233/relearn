#ifndef RELEARN_DOUBLE_Q_LEARNING_HPP
#define RELEARN_DOUBLE_Q_LEARNING_HPP
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
 * @file relearn/double_q_learning.hpp
 * @brief Double Q-Learning (`double_q_learning`)
 */
#include "core.hpp"
#include "schedules.hpp"
#include <deque>
#include <functional>
#include <random>
#include <tuple>

namespace relearn
{
/**
 * @struct double_q_learning This is the Double Q-learning algorithm (Sutton & Barto, Section 6.7)
 * @brief Double Q-learning updates policy values using episodes (`markov_chain`)
 * @param gamma is the discount rate
 * @param alpha is the learning rate
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * Q-learning uses the same values both to pick the best next action and to value it.
 * When those values are noisy, the maximum of them is too high on average: this is
 * the *maximization bias*. Double Q-learning keeps two independent estimates, Q1
 * and Q2. Each update flips a coin, and changes one of them: the estimate being
 * updated picks the best next action, and the *other* one values it
 *
 * Q1(s_t,a_t) = Q1(s_t,a_t) + α * (r_t + γ * Q2(s_{t+1}, argmax_a Q1(s_{t+1}, a)) - Q1(s_t, a_t))
 *
 * (or the same with Q1 and Q2 swapped), and for the last link, which has no next state:
 *
 * Q1(s_t,a_t) = Q1(s_t,a_t) + α * (r_t - Q1(s_t, a_t))
 *
 * The best next action is picked among every action known in s_{t+1}: stored in either
 * estimate, or taken there in the episode, counting 0 for those not stored in the estimate
 * being updated (each update fills in only one estimate).
 *
 * The two estimates are kept in `first` (Q1) and `second` (Q2). The agent acts on
 * both: `policy_map` receives their average, (Q1 + Q2) / 2, for each pair learnt,
 * so being greedy on `policy_map` is being greedy on Q1 + Q2, as in the book.
 */
template <class state_class, 
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double> 
struct double_q_learning
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
     * the state/action pair has been updated *in the estimate being updated*, e.g.
     * `relearn::polynomial_decay<>()` for α = 1/n: each estimate is then the plain average
     * of its own targets. If it is not set, `alpha` is used for every update.
     */
    std::function<value_type(std::size_t)> learning_rate;
    /// the first estimate, Q1
    policy<state_class,action_class,value_type> first;
    /// the second estimate, Q2
    policy<state_class,action_class,value_type> second;
    /// the number of updates of each state/action pair in `first` (counted only when `learning_rate` is set)
    visit_counter<state_class,action_class> first_visits;
    /// the number of updates of each state/action pair in `second` (counted only when `learning_rate` is set)
    visit_counter<state_class,action_class> second_visits;
    /// the coin that chooses which estimate each update changes - seed it to repeat a run
    std::mt19937 coin;

    /**
     * @brief ctor with a learning rate @param initial_alpha, a discount rate @param initial_gamma,
     *        both 0.9 by default, and the @param coin_seed of the coin
     */
    double_q_learning(value_type initial_alpha = 0.9,
                      value_type initial_gamma = 0.9,
                      unsigned int coin_seed = 0);
    /// @brief the update rule of Double Q-learning: updates `first` or `second`,
    ///        and @return their average for the pair
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
double_q_learning<state_class,action_class,markov_chain,value_type
                 >::double_q_learning(value_type initial_alpha,
                                      value_type initial_gamma,
                                      unsigned int coin_seed)
: alpha(initial_alpha),
  gamma(initial_gamma),
  coin(coin_seed)
{}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
typename double_q_learning<state_class,action_class,markov_chain,value_type>::triplet
            double_q_learning<state_class,action_class,markov_chain,value_type
                             >::q_value(markov_chain & episode, 
                                        std::size_t index,
                                        policy<state_class,action_class,value_type> &) 
{
    auto step = episode[index];
    // flip the coin: update Q1 with Q2's opinion, or Q2 with Q1's
    std::bernoulli_distribution heads(0.5);
    const bool update_first = heads(coin);
    auto & updated = update_first ? first : second;
    auto & other   = update_first ? second : first;
    // α: `alpha`, or the scheduled learning rate of this update of (s_t, a_t) in that estimate
    auto & visits = update_first ? first_visits : second_visits;
    auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
    // Q(s_t, a_t) of the estimate we update, and r_t, the reward received after a_t
    auto q = updated.value(step.state, step.action);
    auto r = step.state.reward();
    // the target: r_t, plus for a step that is not the last one, the value of the next state
    value_type target = r;
    if (index + 1 < episode.size()) {
        auto next = episode[index + 1];
        // the actions known in s_{t+1}: stored in either estimate, or taken there in this
        // episode. Each update fills in only one estimate, so either may lack some of them
        auto known = first.actions(next.state);
        for (const auto & item : second.actions(next.state)) {
            known.emplace(item.first, item.second);
        }
        known.emplace(next.action, value_type(0));
        // the estimate we update picks the best next action (0 for actions it has not stored)...
        const action_class * best = nullptr;
        value_type best_value = 0;
        for (const auto & item : known) {
            auto value = updated.value(next.state, item.first);
            if (!best || value > best_value) {
                best = &item.first;
                best_value = value;
            }
        }
        // ...and the other estimate says what it is worth
        target += gamma * other.value(next.state, *best);
    }
    updated.update(step.state, step.action, q + rate * (target - q));
    // the agent acts on both estimates: give it their average
    auto average = (first.value(step.state, step.action) + 
                    second.value(step.state, step.action)) / value_type(2);
    return std::make_tuple(step.state, step.action, average);
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
void double_q_learning<state_class,action_class,markov_chain,value_type
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
