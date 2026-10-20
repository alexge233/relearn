#ifndef RELEARN_DYNAMIC_PROGRAMMING_HPP
#define RELEARN_DYNAMIC_PROGRAMMING_HPP
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
 * @file relearn/dynamic_programming.hpp
 * @brief dynamic programming with a model of the world (`finite_mdp`): policy evaluation,
 *        policy iteration and value iteration (`dynamic_programming`)
 */
#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

namespace relearn
{
/**
 * @struct finite_mdp
 * @brief a model of a finite MDP: its states, the actions of each state, and what each action leads to
 *
 * The model is the four-argument function p(s', r | s, a) of Sutton & Barto, Section 3.1:
 * the probability that taking action a in state s gives the reward r and leads to state s'.
 * It is given as three things:
 *
 * - `states`: every state, terminal ones included (S+ in the book)
 * - `actions(s)`: the actions of state s, A(s); none for a terminal state, whose value is 0
 * - `outcomes(s, a)`: every (s', r) that a can lead to from s, with its probability
 *   p(s', r | s, a); the probabilities of each (s, a) add up to 1
 *
 * Every next state must be one of `states`.
 */
template <class state_class,
          class action_class,
          typename value_type = double>
struct finite_mdp
{
    /// one outcome of taking an action: the next state s', the reward r, and p(s', r | s, a)
    struct outcome
    {
        /// s': the state reached
        state_class next;
        /// r: the reward received
        value_type reward;
        /// p(s', r | s, a): the probability of this outcome
        value_type probability;
    };
    /// S+: every state, terminal ones included
    std::vector<state_class> states;
    /// A(s): the actions of a state - none for a terminal state
    std::function<std::vector<action_class>(const state_class &)> actions;
    /// the outcomes of taking an action in a state, with their probabilities p(s', r | s, a)
    std::function<std::vector<outcome>(const state_class &,
                                       const action_class &)> outcomes;
};

/**
 * @struct dynamic_programming This is dynamic programming (Sutton & Barto, Chapter 4)
 * @brief works out the values of a policy, and the optimal policy, from a model of the world
 * @param gamma is the discount rate
 * @param theta is θ, the accuracy: the sweeps stop once no value changes by θ or more
 *
 * Unlike the learners, dynamic programming does not play: it is given the model,
 * p(s', r | s, a) (a `finite_mdp`), and sweeps over every state, again and again,
 * updating its value V(s) from the values of the states that can follow it.
 * Every update looks one step ahead, at the value of taking action a in s:
 *
 * q(s,a) = Σ_{s',r} p(s',r | s,a) * (r + γ * V(s'))
 *
 * - `evaluate` is iterative policy evaluation (Section 4.1): the values of a policy π,
 *   V(s) = Σ_a π(a | s) * q(s,a)
 * - `improve` is policy improvement (Section 4.2): the greedy policy on V,
 *   π(s) = argmax_a q(s,a)
 * - `policy_iteration` (Section 4.3) evaluates and improves a policy until it is stable
 * - `value_iteration` (Section 4.4) sweeps with V(s) = max_a q(s,a), then reads the greedy policy
 *
 * The sweeps update V in place, as in the book's boxes: each update uses the newest values,
 * some of them from this very sweep. Set `in_place` to false for the "two-array" version,
 * where every update of a sweep uses the values of the sweep before (as in Figure 4.1).
 * The values are kept in `values`, and a state not in it is worth 0.
 *
 * Two departures from the book, to keep it safe to use:
 * - many actions can be equally good, and rounding makes their values differ in the last digits,
 *   so actions whose values are within `ties` of the best are all taken as best: the policy's
 *   current action is kept if it is one of them, otherwise the first of them in A(s) is taken.
 *   Without this, policy iteration can switch between equally good actions for ever
 *   (Sutton & Barto, Exercise 4.4)
 * - the sweeps stop after `max_sweeps`, even if the values still change by more than θ,
 *   e.g. for the values of a policy that never ends an episode, with γ = 1
 */
template <class state_class,
          class action_class,
          typename value_type = double>
struct dynamic_programming
{
    /// the model: p(s', r | s, a)
    using model_type = finite_mdp<state_class,action_class,value_type>;
    /// V: the value of each state
    using values_type = std::unordered_map<state_class,
                                           value_type,
                                           hasher<state_class>>;
    /// a deterministic policy, π(s): the action taken in each state
    using policy_type = std::unordered_map<state_class,
                                           action_class,
                                           hasher<state_class>>;
    /// a stochastic policy, π(a | s): the probability of taking action a in state s
    using probabilities = std::function<value_type(const state_class &,
                                                   const action_class &)>;

    /// discount rate
    value_type gamma = 0.9;
    /// θ: the sweeps stop once no value changes by θ or more
    value_type theta = 1e-9;
    /// actions whose values are within this of the best are equally good
    value_type ties = value_type(1e-9);
    /// true to update the values in place (the default, as in the book), false for the two-array version
    bool in_place = true;
    /// the most sweeps of one evaluation, or of value iteration
    std::size_t max_sweeps = 100000;
    /// V(s): the value of each state, 0 for a state not in it - you may set it to start from other values
    values_type values;

    /**
     * @brief ctor with a discount rate @param initial_gamma (0.9 by default),
     *        and the accuracy @param initial_theta θ (1e-9 by default)
     */
    dynamic_programming(value_type initial_gamma = 0.9,
                        value_type initial_theta = 1e-9);

    /// @return V(@param s), or 0 if it has no value yet
    value_type value(const state_class & s) const;
    /// @return q(s,a) = Σ p(s',r | s,a) * (r + γ * V(s')), for action @param a in state @param s
    value_type q_value(const model_type & model,
                       const state_class & s,
                       const action_class & a) const;

    /**
     * @brief iterative policy evaluation (Section 4.1): the values of the stochastic policy @param pi
     * @param model is the model of the world
     * @param pi gives π(a | s), the probability of taking a in s
     * The values are in `values` afterwards.
     */
    void evaluate(const model_type & model,
                  const probabilities & pi);
    /// @brief iterative policy evaluation of the deterministic policy @param pi, in the @param model
    void evaluate(const model_type & model,
                  const policy_type & pi);

    /**
     * @brief policy improvement (Section 4.2): make @param pi greedy on the values
     * @param model is the model of the world
     * @return true if the policy is stable: no action changed
     * A state with no action in @param pi gets one.
     */
    bool improve(const model_type & model,
                 policy_type & pi) const;
    /// @return the greedy policy on the values, in the @param model
    policy_type greedy(const model_type & model) const;

    /**
     * @brief policy iteration (Section 4.3)
     * @param model is the model of the world
     * @param pi is the policy to start from; a state with no action gets the first one in A(s)
     * @return the optimal policy; the optimal values are in `values`
     */
    policy_type policy_iteration(const model_type & model,
                                 policy_type pi = policy_type());
    /**
     * @brief value iteration (Section 4.4)
     * @param model is the model of the world
     * @return the optimal policy, greedy on the optimal values, which are in `values`
     */
    policy_type value_iteration(const model_type & model);

    /// @brief write q(s,a) of every state and action of the @param model into @param policy_map
    void action_values(const model_type & model,
                       policy<state_class,action_class,value_type> & policy_map) const;

    /// @return the sweeps over the states of the last evaluation or value iteration (all of them, for policy iteration)
    std::size_t sweeps() const;
    /// @return the policy evaluations of the last policy iteration
    std::size_t iterations() const;

private:
    // @return Σ p(s',r | s,a) * (r + γ * V(s')), reading V(s') from @param from
    value_type backup(const model_type & model,
                      const values_type & from,
                      const state_class & s,
                      const action_class & a) const;
    // @return V(s) in @param from, or 0 if it has no value yet
    static value_type lookup(const values_type & from,
                             const state_class & s);
    // @return the best action in @param s, keeping @param current if it is one of the best
    action_class best_action(const model_type & model,
                             const state_class & s,
                             const action_class * current) const;

    // the sweeps of the last evaluation or value iteration (or of all of a policy iteration)
    std::size_t sweeps_ = 0;
    // the policy evaluations of the last policy iteration
    std::size_t iterations_ = 0;
};

/********************************************************************************
 ***********               Template Implementations
 ********************************************************************************/

template <class state_class,
          class action_class,
          typename value_type>
dynamic_programming<state_class,action_class,value_type
                   >::dynamic_programming(value_type initial_gamma,
                                          value_type initial_theta)
: gamma(initial_gamma),
  theta(initial_theta)
{}

template <class state_class,
          class action_class,
          typename value_type>
value_type dynamic_programming<state_class,action_class,value_type
                              >::lookup(const values_type & from,
                                        const state_class & s)
{
    auto found = from.find(s);
    return found == from.end() ? value_type(0) : found->second;
}

template <class state_class,
          class action_class,
          typename value_type>
value_type dynamic_programming<state_class,action_class,value_type
                              >::value(const state_class & s) const
{
    return lookup(values, s);
}

template <class state_class,
          class action_class,
          typename value_type>
value_type dynamic_programming<state_class,action_class,value_type
                              >::backup(const model_type & model,
                                        const values_type & from,
                                        const state_class & s,
                                        const action_class & a) const
{
    // Σ_{s',r} p(s',r | s,a) * (r + γ * V(s'))
    value_type q = 0;
    for (const auto & o : model.outcomes(s, a)) {
        q += o.probability * (o.reward + gamma * lookup(from, o.next));
    }
    return q;
}

template <class state_class,
          class action_class,
          typename value_type>
value_type dynamic_programming<state_class,action_class,value_type
                              >::q_value(const model_type & model,
                                         const state_class & s,
                                         const action_class & a) const
{
    return backup(model, values, s, a);
}

template <class state_class,
          class action_class,
          typename value_type>
void dynamic_programming<state_class,action_class,value_type
                        >::evaluate(const model_type & model,
                                    const probabilities & pi)
{
    for (sweeps_ = 0; sweeps_ < max_sweeps; ) {
        // the two-array version reads every value from the sweep before
        const values_type before = in_place ? values_type() : values;
        const values_type & from = in_place ? values : before;
        // Δ: the largest change of a value in this sweep
        value_type delta = 0;
        for (const auto & s : model.states) {
            // V(s) = Σ_a π(a | s) Σ_{s',r} p(s',r | s,a) * (r + γ * V(s')), and 0 if s is terminal
            value_type v = 0;
            for (const auto & a : model.actions(s)) {
                auto p = pi(s, a);
                // an action π never takes adds nothing
                if (p != 0) {
                    v += p * backup(model, from, s, a);
                }
            }
            delta = std::max(delta, static_cast<value_type>(std::abs(v - lookup(values, s))));
            values[s] = v;
        }
        sweeps_++;
        // stop once no value changed by θ or more
        if (delta < theta) {
            break;
        }
    }
}

template <class state_class,
          class action_class,
          typename value_type>
void dynamic_programming<state_class,action_class,value_type
                        >::evaluate(const model_type & model,
                                    const policy_type & pi)
{
    // π(a | s) is 1 for the action of π(s), and 0 for the others
    evaluate(model, [&pi](const state_class & s,
                          const action_class & a) {
        auto found = pi.find(s);
        return (found != pi.end() && found->second == a) ? value_type(1) : value_type(0);
    });
}

template <class state_class,
          class action_class,
          typename value_type>
action_class dynamic_programming<state_class,action_class,value_type
                                >::best_action(const model_type & model,
                                               const state_class & s,
                                               const action_class * current) const
{
    // q(s,a) of every action, and the best of them
    const auto actions = model.actions(s);
    std::vector<value_type> q;
    for (const auto & a : actions) {
        q.push_back(q_value(model, s, a));
    }
    const value_type best = *std::max_element(q.begin(), q.end());
    // the current action is kept if it is as good as the best, within `ties`
    for (std::size_t i = 0; i < actions.size(); i++) {
        if (current && actions[i] == *current && q[i] >= best - ties) {
            return actions[i];
        }
    }
    // otherwise the first action as good as the best
    for (std::size_t i = 0; i < actions.size(); i++) {
        if (q[i] >= best - ties) {
            return actions[i];
        }
    }
    return actions.front();
}

template <class state_class,
          class action_class,
          typename value_type>
bool dynamic_programming<state_class,action_class,value_type
                        >::improve(const model_type & model,
                                   policy_type & pi) const
{
    bool stable = true;
    for (const auto & s : model.states) {
        // a terminal state has no action to take
        if (model.actions(s).empty()) {
            continue;
        }
        // π(s) = argmax_a Σ_{s',r} p(s',r | s,a) * (r + γ * V(s'))
        auto found = pi.find(s);
        const action_class * old_action = found == pi.end() ? nullptr : &found->second;
        auto best = best_action(model, s, old_action);
        // the policy is stable if no action changed
        if (!old_action || !(*old_action == best)) {
            stable = false;
            pi.erase(s);
            pi.emplace(s, best);
        }
    }
    return stable;
}

template <class state_class,
          class action_class,
          typename value_type>
typename dynamic_programming<state_class,action_class,value_type>::policy_type
    dynamic_programming<state_class,action_class,value_type
                       >::greedy(const model_type & model) const
{
    policy_type pi;
    improve(model, pi);
    return pi;
}

template <class state_class,
          class action_class,
          typename value_type>
typename dynamic_programming<state_class,action_class,value_type>::policy_type
    dynamic_programming<state_class,action_class,value_type
                       >::policy_iteration(const model_type & model,
                                           policy_type pi)
{
    // 1. initialisation: a state without an action takes its first one
    for (const auto & s : model.states) {
        auto actions = model.actions(s);
        if (!actions.empty() && pi.find(s) == pi.end()) {
            pi.emplace(s, actions.front());
        }
    }
    std::size_t all_sweeps = 0;
    iterations_ = 0;
    for (;;) {
        // 2. policy evaluation: V = v_π, starting from the values of the policy before
        evaluate(model, pi);
        all_sweeps += sweeps_;
        iterations_++;
        // 3. policy improvement: greedy on V, until no action changes
        if (improve(model, pi)) {
            break;
        }
    }
    sweeps_ = all_sweeps;
    return pi;
}

template <class state_class,
          class action_class,
          typename value_type>
typename dynamic_programming<state_class,action_class,value_type>::policy_type
    dynamic_programming<state_class,action_class,value_type
                       >::value_iteration(const model_type & model)
{
    for (sweeps_ = 0; sweeps_ < max_sweeps; ) {
        // the two-array version reads every value from the sweep before
        const values_type before = in_place ? values_type() : values;
        const values_type & from = in_place ? values : before;
        // Δ: the largest change of a value in this sweep
        value_type delta = 0;
        for (const auto & s : model.states) {
            // V(s) = max_a Σ_{s',r} p(s',r | s,a) * (r + γ * V(s')), and 0 if s is terminal
            const auto actions = model.actions(s);
            value_type v = 0;
            for (std::size_t i = 0; i < actions.size(); i++) {
                auto q = backup(model, from, s, actions[i]);
                if (i == 0 || q > v) {
                    v = q;
                }
            }
            delta = std::max(delta, static_cast<value_type>(std::abs(v - lookup(values, s))));
            values[s] = v;
        }
        sweeps_++;
        // stop once no value changed by θ or more
        if (delta < theta) {
            break;
        }
    }
    // π(s) = argmax_a Σ_{s',r} p(s',r | s,a) * (r + γ * V(s'))
    return greedy(model);
}

template <class state_class,
          class action_class,
          typename value_type>
void dynamic_programming<state_class,action_class,value_type
                        >::action_values(const model_type & model,
                                         policy<state_class,action_class,value_type> & policy_map) const
{
    for (const auto & s : model.states) {
        for (const auto & a : model.actions(s)) {
            policy_map.update(s, a, q_value(model, s, a));
        }
    }
}

template <class state_class,
          class action_class,
          typename value_type>
std::size_t dynamic_programming<state_class,action_class,value_type>::sweeps() const
{
    return sweeps_;
}

template <class state_class,
          class action_class,
          typename value_type>
std::size_t dynamic_programming<state_class,action_class,value_type>::iterations() const
{
    return iterations_;
}
} // end of namespace
#endif
