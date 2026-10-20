#ifndef RELEARN_PRIORITIZED_SWEEPING_HPP
#define RELEARN_PRIORITIZED_SWEEPING_HPP
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
 * @file relearn/prioritized_sweeping.hpp
 * @brief prioritized sweeping (`prioritized_sweeping`)
 */
#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <set>
#include <unordered_map>
#include <vector>

namespace relearn
{
/**
 * @struct prioritized_sweeping This is prioritized sweeping for a deterministic world (Sutton & Barto, Section 8.4)
 * @brief plans like Dyna-Q, but updates first the moves whose values would change the most
 * @param gamma is the discount rate
 * @param alpha is the learning rate
 * @param planning_steps is n, the most updates after each real step
 * @param theta is θ, the smallest change worth queueing
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * Dyna-Q plans with moves picked at random, most of which change nothing (early on,
 * every value is 0). Prioritized sweeping keeps a queue of the moves whose values
 * would change, the largest change first. For each real step (S, A, R, S') it:
 *
 * 1. remembers it in its model: Model(S,A) = R, S' (the last outcome, as in `dyna_q`),
 *    and that (S, A) is a predecessor of S', a move that leads to S'
 * 2. works out how much the value of (S, A) would change, its priority:
 *    P = |R + γ * max(Q(S', a)) - Q(S, A)|
 *    and puts (S, A) in the queue if P > θ
 * 3. plans, up to `planning_steps` times while the queue is not empty: it takes the
 *    first move (S, A) of the queue, learns from what the model says it gives,
 *    Q(S,A) = Q(S,A) + α * (R + γ * max(Q(S', a)) - Q(S,A)),
 *    and then, since the value of S may have changed, works out the priority of
 *    every predecessor (S̄, Ā) of S, P = |R̄ + γ * max(Q(S, a)) - Q(S̄, Ā)|, and queues it if P > θ.
 *
 * As in the book's box, the real step itself is only learnt through the queue.
 * A move made for the first time is stored in the policy at its value (0), so that
 * max(Q(S, a)), which only counts the actions stored, never misses an action worth 0;
 * if that raises the best value of S, the moves leading to S are queued again.
 * When a move already in the queue is queued again, it keeps the higher of its two
 * priorities, so that each move is in the queue once (the book's box does not show it,
 * but the text of Section 8.4 says so). Moves of equal priority leave the
 * queue in the order they joined it.
 *
 * The model, the predecessors and the queue are kept inside the learner.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct prioritized_sweeping
{
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// n: the most updates from the queue after each real step
    unsigned int planning_steps = 5;
    /// θ: a move joins the queue only if its value would change by more than this
    value_type theta = 0.0001;

    /**
     * @brief ctor with a learning rate @param initial_alpha and a discount rate @param initial_gamma,
     *        both 0.9 by default, the @param initial_planning_steps n (5 by default),
     *        and the threshold @param initial_theta θ (0.0001 by default)
     */
    prioritized_sweeping(value_type initial_alpha = 0.9,
                         value_type initial_gamma = 0.9,
                         unsigned int initial_planning_steps = 5,
                         value_type initial_theta = 0.0001);
    /// @brief remember in the model, and in the predecessors, what happened in the real step at @param index
    void remember(markov_chain & episode,
                  unsigned int index);
    /// @brief queue the move @param a_t in @param s_t, with @param priority P, if P > θ
    void queue(const state_class & s_t,
               const action_class & a_t,
               value_type priority);
    /// @brief learn from up to `planning_steps` moves of the queue - @param policy_map will be modified
    void plan(policy<state_class,action_class,value_type> & policy_map);

    /**
     * @brief do the updating for an episode - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short (e.g., by a step limit): its last
     *        link is then not learnt as the end of the game, and only serves as
     *        the next state of the link before it.
     * Each real step is remembered and queued, and followed by up to `planning_steps` updates.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);

    /// @return the number of real steps learnt so far
    std::size_t steps() const;
    /// @return the number of updates of a value so far (each one is a move taken from the queue)
    std::size_t updates() const;
    /// @return the number of moves in the queue
    std::size_t queued() const;
    /// @return the priority of @param a_t in @param s_t in the queue, or 0 if it is not queued
    value_type priority(const state_class & s_t,
                        const action_class & a_t) const;
    /// @return true if the model remembers really taking @param a_t in @param s_t
    bool remembers(const state_class & s_t,
                   const action_class & a_t) const;
    /// @return the predecessors of @param s_t: the moves the model says lead to it, in the order first made
    std::vector<link<state_class,action_class>> predecessors(const state_class & s_t) const;

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
    };
    // one move in the queue
    struct entry
    {
        // P: how much its value would change
        value_type priority;
        // when it joined the queue: the earlier first, among equal priorities
        std::size_t order;
        // the move (s, a)
        link<state_class,action_class> move;
    };
    // the order of the queue: highest priority first, then the one that joined first
    struct first_out
    {
        bool operator()(const entry & lhs,
                        const entry & rhs) const;
    };
    using queue_type = std::set<entry, first_out>;
    // the entry of each move queued, by its action: the entry itself rather than an iterator
    // into the queue, so that a copy of the learner never points into another learner's queue
    using actions_queued = std::unordered_map<action_class,
                                              entry,
                                              hasher<action_class>>;

    // @return R + γ * max(Q(S', a)) - Q(S, A) for the move @param s_t, @param a_t,
    //         given what the model says it gives, @param happened
    value_type td_error(const state_class & s_t,
                        const action_class & a_t,
                        const outcome & happened,
                        const policy<state_class,action_class,value_type> & policy_map) const;
    // @brief queue every move that leads to @param s_t, with priority |R̄ + γ * max(Q(s_t, a)) - Q(S̄, Ā)|
    void queue_predecessors(const state_class & s_t,
                            const policy<state_class,action_class,value_type> & policy_map);
    // @brief take the move @param s_t, @param a_t out of the predecessors of @param next
    void forget_predecessor(const state_class & s_t,
                            const action_class & a_t,
                            const state_class & next);

    // the model: [state][action] => the last outcome
    std::unordered_map<state_class,
                       std::unordered_map<action_class,
                                          outcome,
                                          hasher<action_class>>,
                       hasher<state_class>> model_;
    // the predecessors: [state] => the moves that lead to it, in the order first made
    std::unordered_map<state_class,
                       std::vector<link<state_class,action_class>>,
                       hasher<state_class>> predecessors_;
    // the queue (PQueue in the book), and where each move queued is in it
    queue_type queue_;
    std::unordered_map<state_class,
                       actions_queued,
                       hasher<state_class>> where_;
    // the number of moves that have joined the queue so far (to order equal priorities)
    std::size_t joined_ = 0;
    // the number of real steps learnt so far
    std::size_t steps_ = 0;
    // the number of updates so far
    std::size_t updates_ = 0;
};

/********************************************************************************
 ***********               Template Implementations
 ********************************************************************************/

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
prioritized_sweeping<state_class,action_class,markov_chain,value_type
                    >::prioritized_sweeping(value_type initial_alpha,
                                            value_type initial_gamma,
                                            unsigned int initial_planning_steps,
                                            value_type initial_theta)
: alpha(initial_alpha),
  gamma(initial_gamma),
  planning_steps(initial_planning_steps),
  theta(initial_theta)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
bool prioritized_sweeping<state_class,action_class,markov_chain,value_type
                         >::first_out::operator()(const entry & lhs,
                                                  const entry & rhs) const
{
    if (lhs.priority != rhs.priority) {
        return lhs.priority > rhs.priority;
    }
    return lhs.order < rhs.order;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void prioritized_sweeping<state_class,action_class,markov_chain,value_type
                         >::remember(markov_chain & episode,
                                     unsigned int index)
{
    auto step = episode[index];
    // what happened: the reward, and the state reached (the last link of an episode ends it)
    const bool terminal = index == episode.size() - 1;
    outcome happened = {static_cast<value_type>(step.state.reward()),
                        terminal ? step.state : episode[index + 1].state,
                        terminal};
    // a move made before, which led somewhere else: it is no longer a predecessor of that state
    auto & actions = model_[step.state];
    auto known = actions.find(step.action);
    if (known != actions.end() && !known->second.terminal) {
        if (happened.terminal || !(known->second.next == happened.next)) {
            forget_predecessor(step.state, step.action, known->second.next);
        }
    }
    // Model(S,A) = R, S' (replacing whatever it remembered before)
    if (known == actions.end()) {
        actions.emplace(step.action, happened);
    }
    else {
        known->second = happened;
    }
    // (S, A) is a predecessor of S', unless the episode ended
    if (!happened.terminal) {
        auto & leading = predecessors_[happened.next];
        link<state_class,action_class> move = {step.state, step.action};
        if (std::find(leading.begin(), leading.end(), move) == leading.end()) {
            leading.push_back(move);
        }
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void prioritized_sweeping<state_class,action_class,markov_chain,value_type
                         >::forget_predecessor(const state_class & s_t,
                                               const action_class & a_t,
                                               const state_class & next)
{
    auto found = predecessors_.find(next);
    if (found == predecessors_.end()) {
        return;
    }
    auto & leading = found->second;
    link<state_class,action_class> move = {s_t, a_t};
    leading.erase(std::remove(leading.begin(), leading.end(), move), leading.end());
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type prioritized_sweeping<state_class,action_class,markov_chain,value_type
                               >::td_error(const state_class & s_t,
                                           const action_class & a_t,
                                           const outcome & happened,
                                           const policy<state_class,action_class,value_type> & policy_map) const
{
    // the target: R, and γ * max(Q(S', a)) unless the episode ended
    value_type target = happened.reward;
    if (!happened.terminal) {
        auto q_next = policy_map.best_value(happened.next);
        if (std::isnan(q_next)) q_next = 0.;
        target += gamma * q_next;
    }
    // R + γ * max(Q(S', a)) - Q(S, A)
    return target - policy_map.value(s_t, a_t);
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void prioritized_sweeping<state_class,action_class,markov_chain,value_type
                         >::queue(const state_class & s_t,
                                  const action_class & a_t,
                                  value_type priority)
{
    // a change of θ or less is not worth an update
    if (!(priority > theta)) {
        return;
    }
    // a move already queued keeps the higher of its two priorities
    auto & actions = where_[s_t];
    auto queued = actions.find(a_t);
    if (queued != actions.end()) {
        if (queued->second.priority >= priority) {
            return;
        }
        queue_.erase(queued->second);
        actions.erase(queued);
    }
    link<state_class,action_class> move = {s_t, a_t};
    const entry joining = {priority, joined_++, move};
    queue_.insert(joining);
    actions.emplace(a_t, joining);
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void prioritized_sweeping<state_class,action_class,markov_chain,value_type
                         >::plan(policy<state_class,action_class,value_type> & policy_map)
{
    for (unsigned int i = 0; i < planning_steps && !queue_.empty(); i++) {
        // S, A = first(PQueue): the move whose value would change the most
        const auto move = queue_.begin()->move;
        queue_.erase(queue_.begin());
        where_[move.state].erase(move.action);
        // R, S' = Model(S, A)
        const auto & happened = model_.at(move.state).at(move.action);
        // Q(S,A) = Q(S,A) + α * (R + γ * max(Q(S', a)) - Q(S,A))
        auto q = policy_map.value(move.state, move.action);
        policy_map.update(move.state, move.action,
                          q + alpha * td_error(move.state, move.action, happened, policy_map));
        updates_++;
        // the value of S may have changed: so may the values of the moves that lead to it
        queue_predecessors(move.state, policy_map);
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void prioritized_sweeping<state_class,action_class,markov_chain,value_type
                         >::queue_predecessors(const state_class & s_t,
                                               const policy<state_class,action_class,value_type> & policy_map)
{
    auto found = predecessors_.find(s_t);
    if (found == predecessors_.end()) {
        return;
    }
    for (const auto & before : found->second) {
        // P = |R̄ + γ * max(Q(S, a)) - Q(S̄, Ā)|, queued if P > θ
        const auto & leads = model_.at(before.state).at(before.action);
        auto p = std::abs(td_error(before.state, before.action, leads, policy_map));
        queue(before.state, before.action, p);
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void prioritized_sweeping<state_class,action_class,markov_chain,value_type
                         >::operator()(markov_chain episode,
                                       policy<state_class,action_class,value_type> & policy_map,
                                       bool terminated)
{
    // a cut-short episode does not learn its last link: it has no outcome yet
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size()
                                                                   : episode.size() - 1;
    for (std::size_t i = 0; i < learnt; i++) {
        steps_++;
        // Model(S,A) = R, S', and (S, A) is a predecessor of S'
        remember(episode, static_cast<unsigned int>(i));
        const auto & step = episode[i];
        // a move made for the first time is stored at its value (0 unless set before), even when it
        // is never worth an update, so that max(Q(S, a)) counts it: an action worth 0 must not lose
        // to one worth less. If it raises the best value of S, the moves leading to S are queued
        if (policy_map.actions(step.state).count(step.action) == 0) {
            const auto best_before = policy_map.best_value(step.state);
            policy_map.update(step.state, step.action, policy_map.value(step.state, step.action));
            // (with nothing stored before, the best value of S already counted as 0: nothing changes)
            if (policy_map.best_value(step.state) > best_before) {
                queue_predecessors(step.state, policy_map);
            }
        }
        // P = |R + γ * max(Q(S', a)) - Q(S, A)|, queued if P > θ
        const auto & happened = model_.at(step.state).at(step.action);
        queue(step.state, step.action,
              std::abs(td_error(step.state, step.action, happened, policy_map)));
        // then plan, from the queue
        plan(policy_map);
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
std::size_t prioritized_sweeping<state_class,action_class,markov_chain,value_type>::steps() const
{
    return steps_;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
std::size_t prioritized_sweeping<state_class,action_class,markov_chain,value_type>::updates() const
{
    return updates_;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
std::size_t prioritized_sweeping<state_class,action_class,markov_chain,value_type>::queued() const
{
    return queue_.size();
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type prioritized_sweeping<state_class,action_class,markov_chain,value_type
                               >::priority(const state_class & s_t,
                                           const action_class & a_t) const
{
    auto found = where_.find(s_t);
    if (found == where_.end()) {
        return value_type(0);
    }
    auto queued = found->second.find(a_t);
    return queued == found->second.end() ? value_type(0) : queued->second.priority;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
bool prioritized_sweeping<state_class,action_class,markov_chain,value_type
                         >::remembers(const state_class & s_t,
                                      const action_class & a_t) const
{
    auto found = model_.find(s_t);
    return found != model_.end() && found->second.count(a_t) > 0;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
std::vector<link<state_class,action_class>>
    prioritized_sweeping<state_class,action_class,markov_chain,value_type
                        >::predecessors(const state_class & s_t) const
{
    auto found = predecessors_.find(s_t);
    if (found == predecessors_.end()) {
        return {};
    }
    return found->second;
}
} // end of namespace
#endif
