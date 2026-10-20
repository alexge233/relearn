#ifndef RELEARN_ELIGIBILITY_TRACES_HPP
#define RELEARN_ELIGIBILITY_TRACES_HPP
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
 * @file relearn/eligibility_traces.hpp
 * @brief eligibility traces: SARSA(λ) (`sarsa_lambda`), TD(λ) (`td_lambda`) and Watkins's Q(λ) (`watkins_q_lambda`)
 */
#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <vector>

namespace relearn
{
/**
 * @class eligibility_traces
 * @brief the eligibility trace z(s, a) of every state/action pair visited in an episode
 *
 * A trace says how much a pair is to blame for what happens now: it jumps up when the pair
 * is visited, and then fades by γλ at every step. Pairs not visited yet have a trace of 0.
 * The learners keep one, cleared at the start of every episode.
 */
template <class state_class,
          class action_class,
          typename value_type = double>
class eligibility_traces
{
public:
    /// @brief one pair and its trace
    struct entry
    {
        state_class state;
        action_class action;
        value_type trace;
    };

    /// @return z(@param s_t, @param a_t), or 0 if the pair has not been visited
    value_type trace(const state_class & s_t,
                     const action_class & a_t) const;
    /**
     * @brief the pair (@param s_t, @param a_t) is visited:
     *        z = z + 1 (accumulating), or z = 1 if @param replacing (Section 12.7)
     */
    void visit(const state_class & s_t,
               const action_class & a_t,
               bool replacing);
    /// @brief Q(s,a) = Q(s,a) + @param step * z(s,a), for every pair with a trace, in @param policy_map
    void update(policy<state_class,action_class,value_type> & policy_map,
                value_type step) const;
    /**
     * @brief z(s,a) = @param factor * z(s,a), for every pair (γλ, or 0 to cut them all).
     *        A trace that falls to exactly 0 can never change a value again, so it is dropped:
     *        with λ = 0, or after Watkins's cut, the traces do not pile up and slow every step down
     */
    void decay(value_type factor);
    /// @brief forget every trace
    void clear();
    /// @return the pairs with a trace (above 0), in the order they were first visited
    const std::vector<entry> & entries() const;

private:
    // the pairs visited, and their traces
    std::vector<entry> entries_;
};

/**
 * @struct sarsa_lambda This is SARSA(λ) (Sutton & Barto, Section 12.7), and TD(λ) (Section 12.2)
 * @brief SARSA(λ) gives each TD error to every pair visited so far, by how recently and how often it was visited
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param lambda is λ, how slowly the traces fade: 0 is SARSA, 1 is close to Monte Carlo
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * At every step t of the episode, in order:
 *
 * δ = r_t + γ * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)          (δ = r_t - Q(s_t, a_t) at the end)
 * z(s,a) = γλ * z(s,a) for every pair, then z(s_t,a_t) = z(s_t,a_t) + 1 (or = 1, if replacing)
 * Q(s,a) = Q(s,a) + α * δ * z(s,a) for every pair
 *
 * with every trace z starting at 0. With a single action per state (a prediction task,
 * as `games::random_walk`) this is TD(λ), V(s) being the value of (s, that action): `td_lambda`.
 *
 * Each call learns one whole episode: the traces start at 0, and are kept in `traces` afterwards,
 * so that you can have a look at them, until the next call. So learn after the episode
 * (`games::play` then the learner), not with `games::play_online`, whose one step at a time
 * would start the traces again at every step. An episode cut short (`terminated` false) ends
 * with the state reached and the action chosen next, which is not learnt as a step.
 *
 * Offline (`offline` true), every δ of the episode is worked out with the values from before
 * the episode, and they change only at its end. With accumulating traces, the changes then add
 * up to exactly those of the λ-return (Section 12.1): every V(s_t) moving by α (G^λ_t - V(s_t)),
 * all of them worked out with the values from before the episode (the equivalence of the forward
 * and backward views, in the book's first edition, Section 7.4).
 * Do bear in mind that the second edition's offline λ-return algorithm (equation 12.4) makes those updates
 * one after the other instead, so a state visited twice in an episode learns a little differently.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct sarsa_lambda
{
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// λ: how slowly the traces fade, 0 ... 1 (0 is SARSA)
    value_type lambda = 0.9;
    /// replacing traces (z = 1 on a visit) instead of accumulating ones (z = z + 1)
    bool replacing = false;
    /// learn offline: the values change only at the end of the episode
    bool offline = false;
    /// the traces z(s, a) of the last episode learnt
    eligibility_traces<state_class,action_class,value_type> traces;

    /**
     * @brief ctor with a learning rate @param initial_alpha, a discount rate @param initial_gamma
     *        and @param initial_lambda, all 0.9 by default, and @param initial_replacing (false by default)
     */
    sarsa_lambda(value_type initial_alpha = 0.9,
                 value_type initial_gamma = 0.9,
                 value_type initial_lambda = 0.9,
                 bool initial_replacing = false);

    /**
     * @brief learn from a whole episode - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short (e.g., by a step limit): its last
     *        link is then not learnt, and only gives the next state and action of the step before it.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);
};

/**
 * @brief TD(λ) (Sutton & Barto, Section 12.2): `sarsa_lambda` for a prediction task
 *
 * relearn keeps values for state/action pairs, so a prediction task has a single action per
 * state, and V(s) is the value of (s, that action). TD(λ)'s update,
 *
 * δ = r_t + γ * V(s_{t+1}) - V(s_t),  z(s) = γλ * z(s) + [s = s_t],  V(s) = V(s) + α * δ * z(s)
 *
 * is then exactly SARSA(λ)'s.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
using td_lambda = sarsa_lambda<state_class,action_class,markov_chain,value_type>;

/**
 * @struct watkins_q_lambda This is Watkins's Q(λ) (Sutton & Barto, Section 12.10)
 * @brief Q-learning with eligibility traces, which are cut after every move the greedy policy would not make
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param lambda is λ, how slowly the traces fade
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * At every step t of the episode, in order:
 *
 * δ = r_t + γ * max_a Q(s_{t+1}, a) - Q(s_t, a_t)          (δ = r_t - Q(s_t, a_t) at the end)
 * z(s,a) = γλ * z(s,a) for every pair if a_t was greedy, otherwise z(s,a) = 0 (all of them are cut)
 * z(s_t,a_t) = z(s_t,a_t) + 1 (or = 1, if replacing)
 * Q(s,a) = Q(s,a) + α * δ * z(s,a) for every pair
 *
 * Like Q-learning it learns the values of the greedy policy, q*, whatever the agent does:
 * the earlier pairs share in what follows only as long as the agent keeps to the greedy policy.
 * The maximum is over the actions stored for s_{t+1} (0 if there are none), as `q_learning`'s,
 * and a_{t+1} is greedy if its value is at least that maximum (so ties count as greedy);
 * both are read before the values of step t change, as in the book.
 *
 * As `sarsa_lambda`, each call learns one whole episode.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct watkins_q_lambda
{
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// λ: how slowly the traces fade, 0 ... 1 (0 is Q-learning)
    value_type lambda = 0.9;
    /// replacing traces (z = 1 on a visit) instead of accumulating ones (z = z + 1)
    bool replacing = false;
    /// the traces z(s, a) of the last episode learnt
    eligibility_traces<state_class,action_class,value_type> traces;

    /**
     * @brief ctor with a learning rate @param initial_alpha, a discount rate @param initial_gamma
     *        and @param initial_lambda, all 0.9 by default, and @param initial_replacing (false by default)
     */
    watkins_q_lambda(value_type initial_alpha = 0.9,
                     value_type initial_gamma = 0.9,
                     value_type initial_lambda = 0.9,
                     bool initial_replacing = false);

    /**
     * @brief learn from a whole episode - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short (e.g., by a step limit): its last
     *        link is then not learnt, and only gives the next state and action of the step before it.
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
          typename value_type>
value_type eligibility_traces<state_class,action_class,value_type
                             >::trace(const state_class & s_t,
                                      const action_class & a_t) const
{
    for (const auto & e : entries_) {
        if (e.state == s_t && e.action == a_t) {
            return e.trace;
        }
    }
    return value_type(0);
}

template <class state_class,
          class action_class,
          typename value_type>
void eligibility_traces<state_class,action_class,value_type
                       >::visit(const state_class & s_t,
                                const action_class & a_t,
                                bool replacing)
{
    // a pair visited before: its trace goes up by 1, or back to 1
    for (auto & e : entries_) {
        if (e.state == s_t && e.action == a_t) {
            e.trace = replacing ? value_type(1) : e.trace + value_type(1);
            return;
        }
    }
    // a pair visited for the first time this episode: its trace was 0
    entries_.push_back({s_t, a_t, value_type(1)});
}

template <class state_class,
          class action_class,
          typename value_type>
void eligibility_traces<state_class,action_class,value_type
                       >::update(policy<state_class,action_class,value_type> & policy_map,
                                 value_type step) const
{
    // Q(s,a) = Q(s,a) + α δ z(s,a), for every pair with a trace
    for (const auto & e : entries_) {
        auto q = policy_map.value(e.state, e.action);
        policy_map.update(e.state, e.action, q + step * e.trace);
    }
}

template <class state_class,
          class action_class,
          typename value_type>
void eligibility_traces<state_class,action_class,value_type>::decay(value_type factor)
{
    // cutting the traces: every one of them goes
    if (factor == 0) {
        entries_.clear();
        return;
    }
    // z(s,a) = factor * z(s,a), and the traces that fell to exactly 0 go
    for (auto & e : entries_) {
        e.trace *= factor;
    }
    entries_.erase(std::remove_if(entries_.begin(),
                                  entries_.end(),
                                  [](const entry & e) { return e.trace == value_type(0); }),
                   entries_.end());
}

template <class state_class,
          class action_class,
          typename value_type>
void eligibility_traces<state_class,action_class,value_type>::clear()
{
    entries_.clear();
}

template <class state_class,
          class action_class,
          typename value_type>
const std::vector<typename eligibility_traces<state_class,action_class,value_type>::entry> &
    eligibility_traces<state_class,action_class,value_type>::entries() const
{
    return entries_;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
sarsa_lambda<state_class,action_class,markov_chain,value_type
            >::sarsa_lambda(value_type initial_alpha,
                            value_type initial_gamma,
                            value_type initial_lambda,
                            bool initial_replacing)
: alpha(initial_alpha),
  gamma(initial_gamma),
  lambda(initial_lambda),
  replacing(initial_replacing)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void sarsa_lambda<state_class,action_class,markov_chain,value_type
                 >::operator()(markov_chain episode,
                               policy<state_class,action_class,value_type> & policy_map,
                               bool terminated)
{
    // a new episode: every trace starts at 0
    traces.clear();
    // a cut-short episode does not learn its last link: it is not a step
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size()
                                                                   : episode.size() - 1;
    // offline, every δ is worked out with the values from before the episode
    const policy<state_class,action_class,value_type> before = offline ? policy_map
                                                                       : policy<state_class,action_class,value_type>();
    const auto & values = offline ? before : policy_map;
    for (std::size_t t = 0; t < learnt; t++) {
        const auto & step = episode[t];
        // the target r_t + γ Q(s_{t+1}, a_{t+1}), with nothing after the last step of an episode
        value_type target = step.state.reward();
        if (t + 1 < episode.size()) {
            const auto & next = episode[t + 1];
            target += gamma * values.value(next.state, next.action);
        }
        // δ = r_t + γ Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
        value_type delta = target - values.value(step.state, step.action);
        // z = γλ z, then z(s_t, a_t) = z(s_t, a_t) + 1 (accumulating) or 1 (replacing)
        traces.decay(gamma * lambda);
        traces.visit(step.state, step.action, replacing);
        // Q(s,a) = Q(s,a) + α δ z(s,a), for every pair
        traces.update(policy_map, alpha * delta);
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
watkins_q_lambda<state_class,action_class,markov_chain,value_type
                >::watkins_q_lambda(value_type initial_alpha,
                                    value_type initial_gamma,
                                    value_type initial_lambda,
                                    bool initial_replacing)
: alpha(initial_alpha),
  gamma(initial_gamma),
  lambda(initial_lambda),
  replacing(initial_replacing)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void watkins_q_lambda<state_class,action_class,markov_chain,value_type
                     >::operator()(markov_chain episode,
                                   policy<state_class,action_class,value_type> & policy_map,
                                   bool terminated)
{
    // a new episode: every trace starts at 0
    traces.clear();
    // a cut-short episode does not learn its last link: it is not a step
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size()
                                                                   : episode.size() - 1;
    // how much the traces fade before the next step: γλ, or 0 to cut them
    value_type fade = 0;
    for (std::size_t t = 0; t < learnt; t++) {
        const auto & step = episode[t];
        // z = γλ z if a_t was greedy, or z = 0 if it was not (nothing to fade before the first step)
        traces.decay(fade);
        // the target r_t + γ max_a Q(s_{t+1}, a), with nothing after the last step of an episode
        value_type target = step.state.reward();
        bool greedy = true;
        if (t + 1 < episode.size()) {
            const auto & next = episode[t + 1];
            // the best value known in s_{t+1}, 0 if none is known (as `q_learning`)
            auto best = policy_map.best_value(next.state);
            if (std::isnan(best)) {
                best = 0;
            }
            target += gamma * best;
            // a_{t+1} is greedy if no action of s_{t+1} is worth more (read before Q changes)
            greedy = policy_map.value(next.state, next.action) >= best;
        }
        // δ = r_t + γ max_a Q(s_{t+1}, a) - Q(s_t, a_t)
        value_type delta = target - policy_map.value(step.state, step.action);
        // z(s_t, a_t) = z(s_t, a_t) + 1 (accumulating) or 1 (replacing)
        traces.visit(step.state, step.action, replacing);
        // Q(s,a) = Q(s,a) + α δ z(s,a), for every pair
        traces.update(policy_map, alpha * delta);
        // the traces will fade by γλ as long as the agent is greedy, and are cut after an exploring move
        fade = greedy ? gamma * lambda : value_type(0);
    }
}
} // end of namespace
#endif
