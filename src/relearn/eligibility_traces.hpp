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
 * @brief eligibility traces: SARSA(λ) (`sarsa_lambda`), TD(λ) (`td_lambda`), Watkins's Q(λ) (`watkins_q_lambda`),
 *        true online TD(λ) and Sarsa(λ) (`true_online_td_lambda`, `true_online_sarsa_lambda`),
 *        truncated TD(λ) and Sarsa(λ) (`truncated_td_lambda`, `truncated_sarsa_lambda`), and the online
 *        λ-return algorithm (`online_lambda_return`); all of them but Watkins's Q(λ) with a λ and a γ
 *        that may vary from state to state (Section 12.8)
 */
#include "core.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <deque>
#include <functional>
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
    /// @brief z(@param s_t, @param a_t) = z(s_t, a_t) + @param amount (the Dutch trace of `true_online_sarsa_lambda`)
    void add(const state_class & s_t,
             const action_class & a_t,
             value_type amount);
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
 * @brief γ(s_t), the discount of a state (Sutton & Barto, Section 12.8)
 * @return @param gamma_of (@param s_t) if it is set, otherwise the constant @param gamma
 *
 * A γ that varies from state to state: a state with γ = 0 ends the return there, as if the episode
 * had ended (but it goes on), and one with γ = 1 does not discount what follows it at all.
 */
template <class state_class,
          typename value_type>
value_type gamma_at(const std::function<value_type(const state_class &)> & gamma_of,
                    value_type gamma,
                    const state_class & s_t);

/**
 * @brief λ(s_t, a_t), the λ of a state/action pair (Sutton & Barto, Section 12.8)
 * @return @param lambda_of (@param s_t, @param a_t) if it is set, otherwise the constant @param lambda
 *
 * A λ that varies: with λ = 0 at a pair, the returns of the steps before it bootstrap fully from its value,
 * and with λ = 1 they look past it, as Monte Carlo would.
 */
template <class state_class,
          class action_class,
          typename value_type>
value_type lambda_at(const std::function<value_type(const state_class &, const action_class &)> & lambda_of,
                     value_type lambda,
                     const state_class & s_t,
                     const action_class & a_t);

/**
 * @class lambda_return_record
 * @brief the steps of an episode, with all that their truncated λ-returns need
 *
 * Step t holds s_t, a_t and r_t, and what follows it: γ_{t+1} = γ(s_{t+1}), λ_{t+1} = λ(s_{t+1}, a_{t+1}) and
 * V_{t+1}, the value of (s_{t+1}, a_{t+1}) when the agent got there (all three 0 after the last step of an
 * episode). The λ-return of step t truncated at h (Sections 12.3, 12.4 and 12.8) is then, from h backwards:
 *
 * G^λ_{h-1:h} = r_{h-1} + γ_h * V_h
 * G^λ_{t:h}   = r_t + γ_{t+1} * ((1 - λ_{t+1}) * V_{t+1} + λ_{t+1} * G^λ_{t+1:h})
 *
 * With a constant λ and γ this is the book's (1 - λ) Σ_{n=1}^{h-t-1} λ^{n-1} G_{t:t+n} + λ^{h-t-1} G_{t:h},
 * every n-step return G_{t:t+n} bootstrapping from V_{t+n}.
 * `truncated_sarsa_lambda` and `online_lambda_return` keep one for the episode they learn.
 */
template <class state_class,
          class action_class,
          typename value_type = double>
class lambda_return_record
{
public:
    /// @brief one step, and what follows it
    struct entry
    {
        state_class state;
        action_class action;
        value_type reward;
        value_type gamma_next;
        value_type lambda_next;
        value_type value_next;
    };

    /// @brief add the next step @param e of the episode
    void push_back(const entry & e);
    /// @brief forget every step: a new episode
    void clear();
    /// @return the number of steps recorded
    std::size_t size() const;
    /// @return step @param t
    const entry & operator[](std::size_t t) const;
    /**
     * @return G^λ_{t:h} of every step t = @param first ... @param h - 1, in that order
     * @param h is the horizon, at most the number of steps recorded
     */
    std::vector<value_type> returns(std::size_t first,
                                    std::size_t h) const;

private:
    // the steps of the episode
    std::vector<entry> steps_;
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
 * the episode (the values themselves change at every step, but no δ sees those changes, so at the
 * end of the episode they have moved by the sum of the changes). With accumulating traces, they add
 * up to exactly those of the λ-return (Section 12.1): every V(s_t) moving by α (G^λ_t - V(s_t)),
 * all of them worked out with the values from before the episode (the equivalence of the forward
 * and backward views, in the book's first edition, Section 7.4).
 * Do bear in mind that the second edition's offline λ-return algorithm (equation 12.4) makes those updates
 * one after the other instead, so a state visited twice in an episode learns a little differently.
 *
 * λ and γ may vary from state to state (Section 12.8): set `lambda_of` and `gamma_of`. Then
 * δ = r_t + γ(s_{t+1}) * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t), and the traces fade by γ(s_t) λ(s_t, a_t) at step t,
 * and offline it is still exactly the λ-return, the one of Section 12.8.
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
    /// optional: λ(s, a), a λ for every state/action pair (Section 12.8), instead of `lambda`
    std::function<value_type(const state_class &, const action_class &)> lambda_of;
    /// optional: γ(s), a discount for every state (Section 12.8), instead of `gamma`
    std::function<value_type(const state_class &)> gamma_of;
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

/**
 * @struct true_online_sarsa_lambda This is true online Sarsa(λ) (Sutton & Barto, Section 12.7),
 *         and true online TD(λ) (Section 12.5)
 * @brief Sarsa(λ) with Dutch traces, which learns after every step exactly what the online λ-return algorithm
 *        (Section 12.4) would have learnt by then
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param lambda is λ, how slowly the traces fade: 0 is SARSA
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * At every step t of the episode, in order, with Q = Q(s_t, a_t) and Q' = Q(s_{t+1}, a_{t+1})
 * read before the step changes anything (Q' = 0 after the last step of an episode):
 *
 * δ = r_t + γ * Q' - Q
 * z(s,a) = γλ * z(s,a) for every pair, then z(s_t,a_t) = z(s_t,a_t) + 1 - αγλ * z_old(s_t,a_t)
 * Q(s,a) = Q(s,a) + α * (δ + Q - Q_old) * z(s,a) for every pair, then Q(s_t,a_t) = Q(s_t,a_t) - α * (Q - Q_old)
 * Q_old = Q'
 *
 * where z_old(s_t,a_t) is the trace of the pair before it fades, and every trace z, and Q_old,
 * start at 0 in every episode. This is the book's algorithm with one feature per state/action
 * pair (tabular): the trace is a *Dutch* trace, between accumulating and replacing.
 *
 * What makes it *true online*: TD(λ) (`sarsa_lambda`) learns roughly what the λ-return algorithm
 * learns, but only exactly so offline, if the values do not change during the episode. The online
 * λ-return algorithm (Section 12.4) works out, after every step h, every update of the episode so far
 * again from its start, each with the λ-return truncated at h, and the values at h are those of the
 * last of them. True online Sarsa(λ) gives the very same values, after every step, at the cost of
 * one step of TD(λ) (van Seijen and Sutton, 2014; van Seijen et al, 2016).
 * With a single action per state (a prediction task, as `games::random_walk`) this is
 * true online TD(λ): `true_online_td_lambda`.
 *
 * It learns online with `games::play_online` (window 1), after every step, or a whole episode
 * at a time after `games::play`, which for a prediction task gives the very same values.
 * Each call learns every step of the episode given, but the last link if the episode is cut short
 * (`terminated` false): that link is the state reached and the action chosen next, which is not a step.
 * To learn an episode a step at a time, start it with `start_episode()`: from then on the traces,
 * and Q_old, are *kept* from one call to the next, until a call ends the episode (`terminated` true).
 * `games::play_online` does this before every game. Any other call is a whole episode on its own,
 * whose traces start at 0, as for `sarsa_lambda`, even after one cut short.
 *
 * λ and γ may vary from state to state (Section 12.8): set `lambda_of` and `gamma_of`. Then δ uses γ(s_{t+1}),
 * and γλ is γ(s_t) λ(s_t, a_t) at step t, in the fading and in the Dutch trace; and it is still exactly
 * the online λ-return algorithm (`online_lambda_return`), with the λ-return of Section 12.8.
 * @warning an episode learnt a step at a time and cut short stays open until the next `start_episode()`
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct true_online_sarsa_lambda
{
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// λ: how slowly the traces fade, 0 ... 1 (0 is SARSA)
    value_type lambda = 0.9;
    /// optional: λ(s, a), a λ for every state/action pair (Section 12.8), instead of `lambda`
    std::function<value_type(const state_class &, const action_class &)> lambda_of;
    /// optional: γ(s), a discount for every state (Section 12.8), instead of `gamma`
    std::function<value_type(const state_class &)> gamma_of;
    /// the Dutch traces z(s, a) of the episode being learnt
    eligibility_traces<state_class,action_class,value_type> traces;
    /// Q_old: the value of the pair reached by the last step learnt, as it was before that step
    value_type q_old = 0;

    /**
     * @brief ctor with a learning rate @param initial_alpha, a discount rate @param initial_gamma
     *        and @param initial_lambda, all 0.9 by default
     */
    true_online_sarsa_lambda(value_type initial_alpha = 0.9,
                             value_type initial_gamma = 0.9,
                             value_type initial_lambda = 0.9);

    /**
     * @brief start a new episode, to be learnt a step at a time: every trace, and Q_old, back to 0,
     *        and kept from one call to the next until a call ends the episode
     */
    void start_episode();

    /**
     * @brief learn the steps of @param episode: the next steps of the episode started with `start_episode()`,
     *        or else a whole episode on its own - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short, or for the last steps of one still being played
     *        (as `games::play_online` does): its last link is then not learnt yet, and only gives the
     *        next state and action of the step before it.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);

private:
    // true from `start_episode()` until a call ends the episode: the calls go on with the same episode
    bool open_ = false;
};

/**
 * @brief true online TD(λ) (Sutton & Barto, Section 12.5): `true_online_sarsa_lambda` for a prediction task
 *
 * As for `td_lambda`, a prediction task has a single action per state, and V(s) is the value of
 * (s, that action). True online TD(λ)'s update,
 *
 * δ = r_t + γ * V' - V,  z(s) = γλ * z(s) + (1 - αγλ * z(s_t)) * [s = s_t],
 * V(s) = V(s) + α * (δ + V - V_old) * z(s) - α * (V - V_old) * [s = s_t],  V_old = V'
 *
 * is then exactly true online Sarsa(λ)'s.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
using true_online_td_lambda = true_online_sarsa_lambda<state_class,action_class,markov_chain,value_type>;

/**
 * @struct truncated_sarsa_lambda This is truncated TD(λ), TTD(λ) (Sutton & Barto, Section 12.3), for state/action
 *         pairs, and for states (`truncated_td_lambda`)
 * @brief each step is learnt n steps after it, from its λ-return truncated there
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param lambda is λ
 * @param n is the horizon: how many steps each update waits, and looks ahead
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * The λ-return of step t needs the whole episode, but it can be cut at a horizon h, the rest of its weight
 * going to the n-step return that reaches h:
 *
 * G^λ_{t:h} = (1 - λ) Σ_{n=1}^{h-t-1} λ^{n-1} G_{t:t+n} + λ^{h-t-1} G_{t:h}
 *
 * At each step h, TTD(λ) learns step t = h - n from it:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (G^λ_{t:h} - Q(s_t,a_t))
 *
 * and once the episode ends (at T), the steps left, in order, from their λ-returns to the end, G^λ_{t:T}.
 * Every n-step return G_{t:t+k} bootstraps from Q(s_{t+k}, a_{t+k}) as it was when the agent got there,
 * before the update of that step, as in the book's way of working it out, so the learner keeps those values.
 *
 * With n = 1 this is SARSA (or TD(0)), whatever λ is; with λ = 1 it is n-step SARSA (`n_step_sarsa`);
 * and with n at least the length of the episode, it is the offline λ-return algorithm of equation 12.4,
 * every update made at the end of the episode, one after the other. In between, it is a cheap way to
 * have most of the λ-return, n steps late.
 *
 * λ and γ may vary from state to state (Section 12.8): set `lambda_of` and `gamma_of`, and the
 * λ-return is the one of Section 12.8 (see `lambda_return_record`).
 *
 * It learns online with `games::play_online` (window 1), or a whole episode at a time after `games::play`.
 * An episode cut short (`terminated` false) ends with the state reached and the action chosen next, which
 * is not a step, and its last n-1 steps are not learnt, as for `n_step_sarsa`. As `true_online_sarsa_lambda`,
 * an episode started with `start_episode()` goes on from one call to the next until a call ends it; any
 * other call is a whole episode on its own.
 * @warning an episode learnt a step at a time and cut short stays open until the next `start_episode()`
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct truncated_sarsa_lambda
{
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// λ, 0 ... 1 (1 is n-step SARSA)
    value_type lambda = 0.9;
    /// n: the horizon, at least 1 (1 is SARSA)
    unsigned int n = 4;
    /// optional: λ(s, a), a λ for every state/action pair (Section 12.8), instead of `lambda`
    std::function<value_type(const state_class &, const action_class &)> lambda_of;
    /// optional: γ(s), a discount for every state (Section 12.8), instead of `gamma`
    std::function<value_type(const state_class &)> gamma_of;

    /**
     * @brief ctor with a learning rate @param initial_alpha, a discount rate @param initial_gamma
     *        and @param initial_lambda, all 0.9 by default, and the horizon @param initial_n (4 by default)
     */
    truncated_sarsa_lambda(value_type initial_alpha = 0.9,
                           value_type initial_gamma = 0.9,
                           value_type initial_lambda = 0.9,
                           unsigned int initial_n = 4);

    /**
     * @brief start a new episode, to be learnt a step at a time: its steps are kept from one call to the next
     *        until a call ends the episode
     */
    void start_episode();

    /**
     * @brief learn the steps of @param episode: the next steps of the episode started with `start_episode()`,
     *        or else a whole episode on its own - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short, or for the last steps of one still being played
     *        (as `games::play_online` does): its last link is then not a step, and only gives the
     *        next state and action of the step before it.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);

private:
    // true from `start_episode()` until a call ends the episode: the calls go on with the same episode
    bool open_ = false;
    // the steps of the episode so far
    lambda_return_record<state_class,action_class,value_type> record_;
    // the steps learnt so far: 0 ... learnt_ - 1
    std::size_t learnt_ = 0;
};

/**
 * @brief truncated TD(λ) (Sutton & Barto, Section 12.3): `truncated_sarsa_lambda` for a prediction task,
 *        with a single action per state, V(s) being the value of (s, that action)
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
using truncated_td_lambda = truncated_sarsa_lambda<state_class,action_class,markov_chain,value_type>;

/**
 * @struct online_lambda_return This is the online λ-return algorithm (Sutton & Barto, Section 12.4)
 * @brief after every step, the whole episode so far is learnt again from its start, each step from its λ-return
 *        truncated at that step
 * @param alpha is the learning rate
 * @param gamma is the discount rate
 * @param lambda is λ
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * After step h - 1 of the episode, the horizon is h. Starting again from the values w_0 the episode
 * started with, every step t = 0 ... h - 1 is learnt in order:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (G^λ_{t:h} - Q(s_t,a_t))
 *
 * and the values at h, w_h, are those of the last of them. Every n-step return G_{t:t+k} bootstraps from
 * Q(s_{t+k}, a_{t+k}) in w_{t+k-1}, the values when the agent got there. At the end of the episode, w_T
 * is the λ-return algorithm learnt online, each update seeing every reward of the episode.
 *
 * It is the best of the λ-return algorithms, and the most costly: horizon h redoes h updates, so an
 * episode of T steps costs about T²/2 of them. True online Sarsa(λ) (`true_online_sarsa_lambda`) gives
 * the very same values after every step, at the cost of one step of Sarsa(λ): use it, and this one to
 * see what it does. With λ = 0 it is SARSA.
 *
 * λ and γ may vary from state to state (Section 12.8): set `lambda_of` and `gamma_of`, and the
 * λ-return is the one of Section 12.8 (see `lambda_return_record`).
 *
 * It learns online with `games::play_online` (window 1), or a whole episode at a time after `games::play`,
 * as `true_online_sarsa_lambda` does, with the same `start_episode()`; the values of the pairs of the
 * episode change after every step, those of the others not at all.
 * @warning an episode learnt a step at a time and cut short stays open until the next `start_episode()`
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct online_lambda_return
{
    /// learning rate - you may change this as you process episodes
    value_type alpha = 0.9;
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// λ, 0 ... 1 (0 is SARSA)
    value_type lambda = 0.9;
    /// optional: λ(s, a), a λ for every state/action pair (Section 12.8), instead of `lambda`
    std::function<value_type(const state_class &, const action_class &)> lambda_of;
    /// optional: γ(s), a discount for every state (Section 12.8), instead of `gamma`
    std::function<value_type(const state_class &)> gamma_of;

    /**
     * @brief ctor with a learning rate @param initial_alpha, a discount rate @param initial_gamma
     *        and @param initial_lambda, all 0.9 by default
     */
    online_lambda_return(value_type initial_alpha = 0.9,
                         value_type initial_gamma = 0.9,
                         value_type initial_lambda = 0.9);

    /**
     * @brief start a new episode, to be learnt a step at a time: its steps, and the values it started with,
     *        are kept from one call to the next until a call ends the episode
     */
    void start_episode();

    /**
     * @brief learn the steps of @param episode: the next steps of the episode started with `start_episode()`,
     *        or else a whole episode on its own - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short, or for the last steps of one still being played
     *        (as `games::play_online` does): its last link is then not a step, and only gives the
     *        next state and action of the step before it.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);

private:
    // a state/action pair of the episode, and its value when the episode started
    struct start_value
    {
        state_class state;
        action_class action;
        value_type value;
    };
    // true from `start_episode()` until a call ends the episode: the calls go on with the same episode
    bool open_ = false;
    // the steps of the episode so far
    lambda_return_record<state_class,action_class,value_type> record_;
    // w_0: the values the pairs of the episode had when it started, in the order they were first visited
    std::vector<start_value> start_;
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
                       >::add(const state_class & s_t,
                              const action_class & a_t,
                              value_type amount)
{
    // a pair with a trace: it goes up by the amount
    for (auto & e : entries_) {
        if (e.state == s_t && e.action == a_t) {
            e.trace += amount;
            return;
        }
    }
    // a pair without one: its trace was 0
    entries_.push_back({s_t, a_t, amount});
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
            target += gamma_at(gamma_of, gamma, next.state) * values.value(next.state, next.action);
        }
        // δ = r_t + γ Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
        value_type delta = target - values.value(step.state, step.action);
        // z = γλ z, then z(s_t, a_t) = z(s_t, a_t) + 1 (accumulating) or 1 (replacing)
        traces.decay(gamma_at(gamma_of, gamma, step.state) * lambda_at(lambda_of, lambda, step.state, step.action));
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

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
true_online_sarsa_lambda<state_class,action_class,markov_chain,value_type
                        >::true_online_sarsa_lambda(value_type initial_alpha,
                                                    value_type initial_gamma,
                                                    value_type initial_lambda)
: alpha(initial_alpha),
  gamma(initial_gamma),
  lambda(initial_lambda)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void true_online_sarsa_lambda<state_class,action_class,markov_chain,value_type>::start_episode()
{
    traces.clear();
    q_old = 0;
    open_ = true;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void true_online_sarsa_lambda<state_class,action_class,markov_chain,value_type
                             >::operator()(markov_chain episode,
                                           policy<state_class,action_class,value_type> & policy_map,
                                           bool terminated)
{
    // the next steps of the episode started with `start_episode()`, or else a whole episode on its own,
    // whose traces and Q_old start at 0
    if (!open_) {
        traces.clear();
        q_old = 0;
    }
    // a cut-short episode does not learn its last link (yet): it is not a step
    const std::size_t learnt = (terminated || episode.size() == 0) ? episode.size()
                                                                   : episode.size() - 1;
    for (std::size_t t = 0; t < learnt; t++) {
        const auto & step = episode[t];
        // Q and Q', read before this step changes anything; nothing after the last step of an episode
        const value_type q = policy_map.value(step.state, step.action);
        value_type q_next = 0;
        value_type discount = 0;
        if (t + 1 < episode.size()) {
            const auto & next = episode[t + 1];
            q_next = policy_map.value(next.state, next.action);
            discount = gamma_at(gamma_of, gamma, next.state);
        }
        // δ = r_t + γ Q' - Q
        const value_type delta = step.state.reward() + discount * q_next - q;
        // γλ at this step: γ(s_t) λ(s_t, a_t)
        const value_type fade = gamma_at(gamma_of, gamma, step.state) * lambda_at(lambda_of, lambda, step.state, step.action);
        // the Dutch trace: z = γλ z, then z(s_t, a_t) = z(s_t, a_t) + 1 - αγλ z_old(s_t, a_t)
        const value_type z_old = traces.trace(step.state, step.action);
        traces.decay(fade);
        traces.add(step.state, step.action, value_type(1) - alpha * fade * z_old);
        // Q(s,a) = Q(s,a) + α (δ + Q - Q_old) z(s,a), for every pair, then Q(s_t,a_t) = Q(s_t,a_t) - α (Q - Q_old)
        traces.update(policy_map, alpha * (delta + q - q_old));
        policy_map.update(step.state, step.action,
                          policy_map.value(step.state, step.action) - alpha * (q - q_old));
        q_old = q_next;
    }
    // the end of the episode: the next call is a new one
    if (terminated) {
        open_ = false;
    }
}

template <class state_class,
          typename value_type>
value_type gamma_at(const std::function<value_type(const state_class &)> & gamma_of,
                    value_type gamma,
                    const state_class & s_t)
{
    return gamma_of ? gamma_of(s_t) : gamma;
}

template <class state_class,
          class action_class,
          typename value_type>
value_type lambda_at(const std::function<value_type(const state_class &, const action_class &)> & lambda_of,
                     value_type lambda,
                     const state_class & s_t,
                     const action_class & a_t)
{
    return lambda_of ? lambda_of(s_t, a_t) : lambda;
}

template <class state_class,
          class action_class,
          typename value_type>
void lambda_return_record<state_class,action_class,value_type>::push_back(const entry & e)
{
    steps_.push_back(e);
}

template <class state_class,
          class action_class,
          typename value_type>
void lambda_return_record<state_class,action_class,value_type>::clear()
{
    steps_.clear();
}

template <class state_class,
          class action_class,
          typename value_type>
std::size_t lambda_return_record<state_class,action_class,value_type>::size() const
{
    return steps_.size();
}

template <class state_class,
          class action_class,
          typename value_type>
const typename lambda_return_record<state_class,action_class,value_type>::entry &
    lambda_return_record<state_class,action_class,value_type>::operator[](std::size_t t) const
{
    return steps_[t];
}

template <class state_class,
          class action_class,
          typename value_type>
std::vector<value_type> lambda_return_record<state_class,action_class,value_type
                                            >::returns(std::size_t first,
                                                       std::size_t h) const
{
    assert(h <= steps_.size() && "the horizon is past the steps recorded");
    std::vector<value_type> G(h > first ? h - first : 0);
    if (G.empty()) {
        return G;
    }
    // G^λ_{h-1:h} = r_{h-1} + γ_h V_h
    const auto & last = steps_[h - 1];
    G.back() = last.reward + last.gamma_next * last.value_next;
    // then backwards: G^λ_{t:h} = r_t + γ_{t+1} ((1 - λ_{t+1}) V_{t+1} + λ_{t+1} G^λ_{t+1:h})
    for (std::size_t i = G.size() - 1; i > 0; i--) {
        const auto & e = steps_[first + i - 1];
        G[i - 1] = e.reward + e.gamma_next * ((1 - e.lambda_next) * e.value_next + e.lambda_next * G[i]);
    }
    return G;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
truncated_sarsa_lambda<state_class,action_class,markov_chain,value_type
                      >::truncated_sarsa_lambda(value_type initial_alpha,
                                                value_type initial_gamma,
                                                value_type initial_lambda,
                                                unsigned int initial_n)
: alpha(initial_alpha),
  gamma(initial_gamma),
  lambda(initial_lambda),
  n(initial_n)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void truncated_sarsa_lambda<state_class,action_class,markov_chain,value_type>::start_episode()
{
    record_.clear();
    learnt_ = 0;
    open_ = true;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void truncated_sarsa_lambda<state_class,action_class,markov_chain,value_type
                           >::operator()(markov_chain episode,
                                         policy<state_class,action_class,value_type> & policy_map,
                                         bool terminated)
{
    assert(n >= 1 && "truncated_sarsa_lambda needs n >= 1");
    const std::size_t horizon = std::max(n, 1u);
    // the next steps of the episode started with `start_episode()`, or else a whole episode on its own
    if (!open_) {
        record_.clear();
        learnt_ = 0;
    }
    // learn step t from G^λ_{t:h}: Q(s_t,a_t) = Q(s_t,a_t) + α (G^λ_{t:h} - Q(s_t,a_t))
    auto learn = [&](std::size_t t, std::size_t h) {
        const auto & e = record_[t];
        const value_type G = record_.returns(t, h).front();
        const value_type q = policy_map.value(e.state, e.action);
        policy_map.update(e.state, e.action, q + alpha * (G - q));
    };
    // a cut-short episode does not learn its last link: it is not a step
    const std::size_t steps = (terminated || episode.size() == 0) ? episode.size()
                                                                  : episode.size() - 1;
    for (std::size_t t = 0; t < steps; t++) {
        const auto & step = episode[t];
        // the step, and what follows it: γ_{t+1}, λ_{t+1} and V_{t+1}, the value of (s_{t+1}, a_{t+1})
        // before the update of this step, or nothing after the last step of an episode
        typename lambda_return_record<state_class,action_class,value_type>::entry e
            {step.state, step.action, step.state.reward(), value_type(0), value_type(0), value_type(0)};
        if (t + 1 < episode.size()) {
            const auto & next = episode[t + 1];
            e.gamma_next  = gamma_at(gamma_of, gamma, next.state);
            e.lambda_next = lambda_at(lambda_of, lambda, next.state, next.action);
            e.value_next  = policy_map.value(next.state, next.action);
        }
        record_.push_back(e);
        // at h = t + 1, the step n steps back is learnt, from its λ-return truncated at h
        const std::size_t h = record_.size();
        while (learnt_ + horizon <= h) {
            learn(learnt_, h);
            learnt_++;
        }
    }
    // the end of the episode: the steps left, in order, from their λ-returns to the end; the next call is a new one
    if (terminated) {
        const std::size_t T = record_.size();
        for (; learnt_ < T; learnt_++) {
            learn(learnt_, T);
        }
        open_ = false;
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
online_lambda_return<state_class,action_class,markov_chain,value_type
                    >::online_lambda_return(value_type initial_alpha,
                                            value_type initial_gamma,
                                            value_type initial_lambda)
: alpha(initial_alpha),
  gamma(initial_gamma),
  lambda(initial_lambda)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void online_lambda_return<state_class,action_class,markov_chain,value_type>::start_episode()
{
    record_.clear();
    start_.clear();
    open_ = true;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void online_lambda_return<state_class,action_class,markov_chain,value_type
                         >::operator()(markov_chain episode,
                                       policy<state_class,action_class,value_type> & policy_map,
                                       bool terminated)
{
    // the next steps of the episode started with `start_episode()`, or else a whole episode on its own
    if (!open_) {
        record_.clear();
        start_.clear();
    }
    // a cut-short episode does not learn its last link: it is not a step
    const std::size_t steps = (terminated || episode.size() == 0) ? episode.size()
                                                                  : episode.size() - 1;
    for (std::size_t t = 0; t < steps; t++) {
        const auto & step = episode[t];
        // a pair new to the episode: its value is still the one the episode started with
        bool known = false;
        for (const auto & p : start_) {
            if (p.state == step.state && p.action == step.action) {
                known = true;
                break;
            }
        }
        if (!known) {
            start_.push_back({step.state, step.action, policy_map.value(step.state, step.action)});
        }
        // the step, and what follows it: γ_{t+1}, λ_{t+1} and V_{t+1}, the value of (s_{t+1}, a_{t+1})
        // in w_t, before this step is learnt, or nothing after the last step of an episode
        typename lambda_return_record<state_class,action_class,value_type>::entry e
            {step.state, step.action, step.state.reward(), value_type(0), value_type(0), value_type(0)};
        if (t + 1 < episode.size()) {
            const auto & next = episode[t + 1];
            e.gamma_next  = gamma_at(gamma_of, gamma, next.state);
            e.lambda_next = lambda_at(lambda_of, lambda, next.state, next.action);
            e.value_next  = policy_map.value(next.state, next.action);
        }
        record_.push_back(e);
        // the horizon h: from w_0, every step k < h learnt again, in order, from G^λ_{k:h}
        const std::size_t h = record_.size();
        const auto G = record_.returns(0, h);
        policy<state_class,action_class,value_type> redone;
        for (const auto & p : start_) {
            redone.update(p.state, p.action, p.value);
        }
        for (std::size_t k = 0; k < h; k++) {
            const auto & s = record_[k];
            const value_type q = redone.value(s.state, s.action);
            redone.update(s.state, s.action, q + alpha * (G[k] - q));
        }
        // w_h: the values of the pairs of the episode
        for (const auto & p : start_) {
            policy_map.update(p.state, p.action, redone.value(p.state, p.action));
        }
    }
    // the end of the episode: the next call is a new one
    if (terminated) {
        open_ = false;
    }
}
} // end of namespace
#endif
