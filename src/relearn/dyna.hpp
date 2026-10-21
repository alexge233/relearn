#ifndef RELEARN_DYNA_HPP
#define RELEARN_DYNA_HPP
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
 * @file relearn/dyna.hpp
 * @brief Dyna-Q and Dyna-Q+ (`dyna_q`)
 */
#include "core.hpp"
#include <cmath>
#include <deque>
#include <functional>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace relearn
{
/**
 * @struct dyna_q This is the Dyna-Q and Dyna-Q+ algorithm (Sutton & Barto, Sections 8.2 and 8.3)
 * @brief Dyna-Q learns from each real step, and then again from steps it imagines with a model
 * @param gamma is the discount rate
 * @param alpha is the learning rate
 * @param planning_steps is n, the number of imagined steps after each real one
 * @param kappa is κ, the exploration bonus of Dyna-Q+ (0 for Dyna-Q)
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * For each real step (s_t, a_t, r_t, s_{t+1}), Dyna-Q does three things:
 *
 * 1. learns from it, as Q-learning does:
 *    Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t + γ * max(Q(s_{t+1}, a)) - Q(s_t, a_t))
 * 2. remembers it in its model: taking a_t in s_t gave r_t and led to s_{t+1}
 *    (the model keeps the last outcome of each pair, so it assumes a deterministic world)
 * 3. plans: `planning_steps` times, it picks a state it has seen and an action it has
 *    taken there, at random, asks the model what happened, and learns from that as in 1.
 *
 * Dyna-Q+ adds a bonus to the reward of each imagined step, κ * sqrt(τ), where τ is the
 * number of real steps since the pair was last tried for real. Pairs not tried for a
 * long time look better and better, so the agent goes back to check whether the
 * world has changed.
 *
 * The model, and the clock τ is counted with, are kept inside the learner.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct dyna_q
{
    using triplet = std::tuple<state_class,
                               action_class,
                               value_type>;
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// n: the number of imagined (planning) steps after each real step - 0 is plain Q-learning
    unsigned int planning_steps = 5;
    /// κ: the exploration bonus of Dyna-Q+ - 0 is Dyna-Q, a small positive value (e.g. 0.001) Dyna-Q+
    value_type kappa = 0;
    /**
     * optional, for Dyna-Q+: every action that can be taken in a state.
     * When it is set (and κ > 0), the actions of a state not tried yet are planned with too,
     * as if they led back to the same state with a reward of 0, so that their bonus grows
     * and the agent goes to try them (Sutton & Barto, the footnote in Section 8.3).
     */
    std::function<std::vector<action_class>(const state_class &)> legal_actions;
    /// picks the remembered state and action of each imagined step - seed it to repeat a run
    std::mt19937 prng;

    /**
     * @brief ctor with a learning rate @param initial_alpha and a discount rate @param initial_gamma,
     *        both 0.9 by default, the @param initial_planning_steps n (5 by default),
     *        and the @param prng_seed of the generator that picks the imagined steps
     */
    dyna_q(value_type initial_alpha = 0.9,
           value_type initial_gamma = 0.9,
           unsigned int initial_planning_steps = 5,
           unsigned int prng_seed = 0);
    /// @brief the update rule of Q-learning, for the real step at @param index
    triplet q_value(markov_chain & episode,
                    std::size_t index,
                    policy<state_class,action_class,value_type> & policy_map);
    /// @brief remember in the model what happened in the real step at @param index
    void remember(markov_chain & episode,
                  std::size_t index);
    /// @brief learn from `planning_steps` imagined steps - @param policy_map will be modified
    void plan(policy<state_class,action_class,value_type> & policy_map);

    /**
     * @brief do the updating for an episode - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short (e.g., by a step limit): its last
     *        link is then not learnt as the end of the game, and only serves as
     *        the next state of the link before it.
     * Each step learnt is followed by `planning_steps` imagined ones.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);

    /// @return the number of real steps learnt so far: the clock of Dyna-Q+'s τ
    std::size_t steps() const;
    /// @return true if the model remembers really taking @param a_t in @param s_t
    bool remembers(const state_class & s_t,
                   const action_class & a_t) const;
private:
    // what the model remembers of one state/action pair: its last real outcome
    struct outcome
    {
        // r: the reward received
        value_type reward;
        // s': the state reached (not used when `terminal`)
        state_class next;
        // true if the episode ended after this step
        bool terminal;
        // the real step at which the pair was last tried (for Dyna-Q+'s τ)
        std::size_t last_tried;
        // false for an action Dyna-Q+ only imagines, since it has never been tried
        bool tried;
    };
    // what the model remembers of one state: the actions taken there, and their outcomes
    struct remembered
    {
        // the actions taken in the state, in the order they were first taken
        std::vector<action_class> taken;
        // the last outcome of each of them
        std::unordered_map<action_class,
                           outcome,
                           hasher<action_class>> outcomes;
    };
    // the model: [state] => the actions taken there and their outcomes
    std::unordered_map<state_class,
                       remembered,
                       hasher<state_class>> model_;
    // the states seen, in the order they were first seen (to pick one at random)
    std::vector<state_class> seen_;
    // the number of real steps learnt so far
    std::size_t steps_ = 0;
};

/********************************************************************************
 ***********               Template Implementations
 ********************************************************************************/

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
dyna_q<state_class,action_class,markov_chain,value_type
      >::dyna_q(value_type initial_alpha,
                value_type initial_gamma,
                unsigned int initial_planning_steps,
                unsigned int prng_seed)
: alpha(initial_alpha),
  gamma(initial_gamma),
  planning_steps(initial_planning_steps),
  prng(prng_seed)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
typename dyna_q<state_class,action_class,markov_chain,value_type>::triplet
            dyna_q<state_class,action_class,markov_chain,value_type
                  >::q_value(markov_chain & episode,
                             std::size_t index,
                             policy<state_class,action_class,value_type> & policy_map)
{
    auto step = episode[index];
    // Q(s_t, a_t), and r_t, the reward received after a_t
    auto q = policy_map.value(step.state, step.action);
    auto r = step.state.reward();
    // the last decision of an episode has no next state to bootstrap from
    if (index + 1 == episode.size()) {
        return std::make_tuple(step.state, step.action, q + alpha * (r - q));
    }
    // Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t + γ * max(Q(s_{t+1}, a)) - Q(s_t, a_t))
    auto q_next = policy_map.best_value(episode[index + 1].state);
    if (std::isnan(q_next)) q_next = 0.;
    return std::make_tuple(step.state, step.action,
                           q + alpha * (r + (gamma * q_next) - q));
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void dyna_q<state_class,action_class,markov_chain,value_type
           >::remember(markov_chain & episode,
                       std::size_t index)
{
    auto step = episode[index];
    // what happened: the reward, and the state reached (the last link of an episode ends it)
    const bool terminal = index + 1 == episode.size();
    outcome happened = {static_cast<value_type>(step.state.reward()),
                        terminal ? step.state : episode[index + 1].state,
                        terminal,
                        steps_,
                        true};
    // a state seen for the first time joins the states imagined steps can start from
    auto found = model_.find(step.state);
    if (found == model_.end()) {
        seen_.push_back(step.state);
        found = model_.emplace(step.state, remembered()).first;
        // Dyna-Q+: its actions not tried yet are imagined to lead back to it, with a reward of 0,
        // as if last tried at step 0 (so their bonus grows from the start)
        if (kappa > 0 && legal_actions) {
            for (const auto & a_t : legal_actions(step.state)) {
                found->second.taken.push_back(a_t);
                found->second.outcomes.emplace(a_t, outcome{0, step.state, false, 0, false});
            }
        }
    }
    // an action taken for the first time in this state joins its actions;
    // one taken before has its outcome replaced by the latest one
    auto & actions = found->second;
    auto known = actions.outcomes.find(step.action);
    if (known == actions.outcomes.end()) {
        actions.taken.push_back(step.action);
        actions.outcomes.emplace(step.action, happened);
    }
    else {
        known->second = happened;
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void dyna_q<state_class,action_class,markov_chain,value_type
           >::plan(policy<state_class,action_class,value_type> & policy_map)
{
    // nothing to imagine before the first real step
    if (seen_.empty()) {
        return;
    }
    for (unsigned int i = 0; i < planning_steps; i++) {
        // a state seen before, at random, and an action taken there before, at random
        std::uniform_int_distribution<std::size_t> pick_state(0, seen_.size() - 1);
        const auto & s_t = seen_[pick_state(prng)];
        const auto & actions = model_.at(s_t);
        std::uniform_int_distribution<std::size_t> pick_action(0, actions.taken.size() - 1);
        const auto & a_t = actions.taken[pick_action(prng)];
        // the model says what happened when a_t was last taken in s_t
        const auto & happened = actions.outcomes.at(a_t);
        // Dyna-Q+: a bonus of κ * sqrt(τ), where τ is the real steps since the pair was last tried
        value_type target = happened.reward;
        if (kappa > 0) {
            target += kappa * std::sqrt(static_cast<value_type>(steps_ - happened.last_tried));
        }
        // learn from the imagined step as from a real one: r + γ * max(Q(s', a))
        if (!happened.terminal) {
            auto q_next = policy_map.best_value(happened.next);
            if (std::isnan(q_next)) q_next = 0.;
            target += gamma * q_next;
        }
        auto q = policy_map.value(s_t, a_t);
        policy_map.update(s_t, a_t, q + alpha * (target - q));
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void dyna_q<state_class,action_class,markov_chain,value_type
           >::operator()(markov_chain episode,
                         policy<state_class,action_class,value_type> & policy_map,
                         bool terminated)
{
    // a cut-short episode does not learn its last link: it has no outcome yet
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size()
                                                               : episode.size() - 1;
    for (std::size_t i = 0; i < learnt; i++) {
        // one more real step on the clock
        steps_++;
        // learn from the real step
        auto update = q_value(episode, i, policy_map);
        policy_map.update(std::get<0>(update),
                          std::get<1>(update),
                          std::get<2>(update));
        // remember it, and then learn from imagined steps
        remember(episode, i);
        plan(policy_map);
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
std::size_t dyna_q<state_class,action_class,markov_chain,value_type>::steps() const
{
    return steps_;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
bool dyna_q<state_class,action_class,markov_chain,value_type
           >::remembers(const state_class & s_t,
                        const action_class & a_t) const
{
    auto found = model_.find(s_t);
    if (found == model_.end()) {
        return false;
    }
    auto known = found->second.outcomes.find(a_t);
    return known != found->second.outcomes.end() && known->second.tried;
}
} // end of namespace
#endif
