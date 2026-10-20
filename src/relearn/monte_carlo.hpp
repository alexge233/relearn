#ifndef RELEARN_MONTE_CARLO_HPP
#define RELEARN_MONTE_CARLO_HPP
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
 * @file relearn/monte_carlo.hpp
 * @brief Monte Carlo control: on-policy (`monte_carlo`) and off-policy (`off_policy_monte_carlo`)
 */
#include "core.hpp"
#include "schedules.hpp"
#include <cassert>
#include <deque>
#include <functional>
#include <unordered_map>
#include <vector>

namespace relearn
{
/**
 * @struct monte_carlo This is on-policy Monte Carlo control (Sutton & Barto, Sections 5.3 and 5.4)
 * @brief Monte Carlo learns from whole episodes: each value moves towards the return that followed it
 * @param gamma is the discount rate
 * @param first_visit is true for first-visit Monte Carlo, false for every-visit
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * Once an episode has ended, the return that followed each step is worked out backwards
 *
 * G_t = r_t + γ * G_{t+1}, with G = 0 after the last step
 *
 * and each value moves towards it, by default as the plain average of all the returns seen:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + 1/n * (G_t - Q(s_t,a_t)), with n the returns averaged so far
 *
 * First-visit Monte Carlo averages, in each episode, only the return after the first time
 * a pair was visited; every-visit Monte Carlo averages the returns after every visit.
 *
 * Monte Carlo needs the real return, so it only learns from episodes that ended:
 * an episode cut short (`terminated` false) is not learnt at all.
 */
template <class state_class, 
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double> 
struct monte_carlo
{
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// true for first-visit Monte Carlo (the default), false for every-visit
    bool first_visit = true;
    /**
     * optional: the learning rate of each update, given the number of times n = 1, 2, ...
     * the state/action pair has been updated, e.g. a constant for a world that changes.
     * If it is not set, the rate is 1/n: each value is the plain average of its returns.
     */
    std::function<value_type(std::size_t)> learning_rate;
    /// the number of updates of each state/action pair (the n of the average)
    visit_counter<state_class,action_class> visits;

    /// ctor with a discount rate @param initial_gamma (0.9 by default) and @param initial_first_visit
    monte_carlo(value_type initial_gamma = 0.9,
                bool initial_first_visit = true);

    /**
     * @brief learn from an episode - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link;
     *        an episode cut short is not learnt, since its return is not known
     */
    void operator()(markov_chain episode, 
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);
};

/**
 * @struct off_policy_monte_carlo This is off-policy Monte Carlo control (Sutton & Barto, Section 5.7)
 * @brief learns the greedy policy from episodes played by another policy, with weighted importance sampling
 * @param gamma is the discount rate
 * @param behaviour gives b(a | s), the probability that the policy which played the episodes took a in s
 *
 * The agent plays by a *behaviour* policy b, which explores (e.g. ε-greedy, or at random),
 * whilst the *target* policy π being learnt is greedy on Q. Walking the episode backwards,
 * each step is weighted by how much more likely π was to take it than b:
 *
 * G = γ * G + r_t
 * C(s_t,a_t) = C(s_t,a_t) + W
 * Q(s_t,a_t) = Q(s_t,a_t) + W / C(s_t,a_t) * (G - Q(s_t,a_t))
 *
 * and then, if a_t is not the greedy action of s_t, π would never have taken it, so the
 * earlier steps tell nothing about π and the walk stops; otherwise W = W / b(a_t | s_t).
 *
 * The greedy action is the action of highest value among those stored for s_t (ties go to
 * the one `policy::best_action` returns). The cumulative weights C are kept in `weights`.
 * Like `monte_carlo`, it only learns from episodes that ended.
 *
 * b(a_t | s_t) is read for every step *before* any value changes, so b may depend on Q
 * (e.g. ε-greedy on the values being learnt): the probabilities used are the ones the episode was played with.
 */
template <class state_class, 
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double> 
struct off_policy_monte_carlo
{
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// b(a | s): the probability that the behaviour policy takes action a in state s - required
    std::function<value_type(const state_class &, const action_class &)> behaviour;
    /// C(s, a): the sum of the importance weights of each state/action pair so far
    policy<state_class,action_class,value_type> weights;

    /**
     * @brief ctor with a discount rate @param initial_gamma (0.9 by default),
     *        and the behaviour policy @param b, giving b(a | s)
     */
    off_policy_monte_carlo(value_type initial_gamma = 0.9,
                           std::function<value_type(const state_class &,
                                                    const action_class &)> b = nullptr);

    /**
     * @brief learn from an episode played by the behaviour policy - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link;
     *        an episode cut short is not learnt, since its return is not known
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
monte_carlo<state_class,action_class,markov_chain,value_type
           >::monte_carlo(value_type initial_gamma,
                          bool initial_first_visit)
: gamma(initial_gamma),
  first_visit(initial_first_visit)
{}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
void monte_carlo<state_class,action_class,markov_chain,value_type
                >::operator()(markov_chain episode, 
                              policy<state_class,action_class,value_type> & policy_map,
                              bool terminated)
{
    // an episode cut short has no return yet: nothing to learn
    if (!terminated || episode.size() == 0) {
        return;
    }
    // for first-visit Monte Carlo: the step at which each pair was first visited
    visit_counter<state_class,action_class> seen;
    std::vector<bool> first(episode.size(), true);
    for (std::size_t t = 0; t < episode.size(); t++) {
        first[t] = seen.increment(episode[t].state, episode[t].action) == 1;
    }
    // walk the episode backwards, adding up the return G_t = r_t + γ G_{t+1}
    value_type G = 0;
    for (std::size_t i = episode.size(); i-- > 0; ) {
        const auto & step = episode[i];
        G = step.state.reward() + gamma * G;
        // first-visit: only the return after the first visit of the pair counts
        if (first_visit && !first[i]) {
            continue;
        }
        // Q(s_t,a_t) = Q(s_t,a_t) + rate * (G_t - Q(s_t,a_t)), with rate = 1/n unless set
        auto n = visits.increment(step.state, step.action);
        auto rate = learning_rate ? learning_rate(n) 
                                  : value_type(1) / static_cast<value_type>(n);
        auto q = policy_map.value(step.state, step.action);
        policy_map.update(step.state, step.action, q + rate * (G - q));
    }
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
off_policy_monte_carlo<state_class,action_class,markov_chain,value_type
                      >::off_policy_monte_carlo(value_type initial_gamma,
                                                std::function<value_type(const state_class &,
                                                                         const action_class &)> b)
: gamma(initial_gamma),
  behaviour(b)
{}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
void off_policy_monte_carlo<state_class,action_class,markov_chain,value_type
                           >::operator()(markov_chain episode, 
                                         policy<state_class,action_class,value_type> & policy_map,
                                         bool terminated)
{
    // an episode cut short has no return yet: nothing to learn
    if (!terminated || episode.size() == 0) {
        return;
    }
    assert(behaviour && "off_policy_monte_carlo needs the behaviour policy b(a | s)");
    // b(a_t | s_t) of every step, read before any value changes: if b depends on Q,
    // these are still the probabilities the episode was played with
    std::vector<value_type> b(episode.size());
    for (std::size_t t = 0; t < episode.size(); t++) {
        b[t] = behaviour(episode[t].state, episode[t].action);
    }
    // walk the episode backwards: the return G, and the importance weight W
    value_type G = 0;
    value_type W = 1;
    for (std::size_t i = episode.size(); i-- > 0; ) {
        const auto & step = episode[i];
        G = step.state.reward() + gamma * G;
        // C(s_t,a_t) = C(s_t,a_t) + W
        auto C = weights.value(step.state, step.action) + W;
        weights.update(step.state, step.action, C);
        // Q(s_t,a_t) = Q(s_t,a_t) + W / C * (G - Q(s_t,a_t))
        auto q = policy_map.value(step.state, step.action);
        policy_map.update(step.state, step.action, q + (W / C) * (G - q));
        // the greedy policy would not have taken a_t: the earlier steps tell nothing about it
        auto greedy = policy_map.best_action(step.state);
        if (!greedy || !(*greedy == step.action)) {
            break;
        }
        // π takes a_t for sure, b with probability b(a_t | s_t): W = W * 1 / b(a_t | s_t)
        if (b[i] <= 0) {
            break;
        }
        W = W / b[i];
    }
}
} // end of namespace
#endif
