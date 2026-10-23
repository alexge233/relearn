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
 * @brief Monte Carlo control: on-policy (`monte_carlo`) and off-policy (`off_policy_monte_carlo`),
 *        and off-policy Monte Carlo prediction with importance sampling (`off_policy_monte_carlo_prediction`)
 */
#include "core.hpp"
#include "schedules.hpp"
#include <cassert>
#include <deque>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace relearn
{
/**
 * @struct monte_carlo This is on-policy Monte Carlo control (Sutton & Barto, Sections 5.3 and 5.4)
 * @brief Monte Carlo learns from whole episodes: each value moves towards the return that followed it
 * @param gamma is the discount rate
 * @param first_visit is true for first-visit Monte Carlo, false for every-visit
 * @warning it learns whole episodes: play them with `games::play`, then learn. `games::play_online`
 *          hands the learner one step at a time, which it ignores, and then the last step alone,
 *          which it would learn as a whole episode
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
 * @warning like `monte_carlo`, it learns whole episodes: use `games::play`, not `games::play_online`
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

/**
 * @brief the importance sampling of `off_policy_monte_carlo_prediction` (Sutton & Barto, Sections 5.5, 5.8 and 5.9)
 */
enum class importance_sampling
{
    /// the plain average of the returns, each weighted by its ratio (equation 5.5): unbiased, but its variance can be huge
    ordinary,
    /// the returns averaged with their ratios as the weights (equation 5.6): biased at first, with a far smaller variance
    weighted,
    /// ordinary, with each return cut into flat partial returns as γ says it is (equation 5.9)
    discounting_aware_ordinary,
    /// weighted, with each return cut into flat partial returns as γ says it is (equation 5.10)
    discounting_aware_weighted,
    /// ordinary, with each reward weighted only by the ratios of the actions before it (equations 5.14 and 5.15)
    per_decision
};

/**
 * @struct off_policy_monte_carlo_prediction This is off-policy Monte Carlo prediction (Sutton & Barto,
 *         Sections 5.5, 5.6, 5.8 and 5.9)
 * @brief learns the values of a target policy π from whole episodes played by another policy b,
 *        with importance sampling: the state values v_π(s) and the action values q_π(s, a)
 * @param gamma is the discount rate
 * @param target gives π(a | s), the probability that the policy being evaluated takes a in s
 * @param behaviour gives b(a | s), the probability that the policy which played the episodes took a in s
 * @param sampling is the importance sampling: `ordinary`, `weighted`, their discounting-aware versions,
 *        or `per_decision`
 * @param first_visit is true for first-visit Monte Carlo, false for every-visit
 * @warning like `monte_carlo`, it learns whole episodes: use `games::play`, not `games::play_online`
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t (the book's R_{t+1}).
 *
 * A return G_t that followed b says how good b is, not π. The importance sampling ratio
 *
 * ρ_{t:h} = Π_{k=t}^{h} π(a_k | s_k) / b(a_k | s_k)
 *
 * is how much more likely π was to take those actions than b, and weighting each return by it
 * turns b's returns into π's. With T(s) the steps at which s was visited (every visit, as in the book's
 * box of Section 5.6, or only the first of each episode if `first_visit` is set), and T the end of each
 * of their episodes,
 *
 * ordinary:  V(s) = Σ_{t in T(s)} ρ_{t:T-1} G_t / |T(s)|
 * weighted:  V(s) = Σ_{t in T(s)} ρ_{t:T-1} G_t / Σ_{t in T(s)} ρ_{t:T-1}   (0 while the sum is 0)
 *
 * Ordinary importance sampling is unbiased, but its variance can be huge, even infinite (Example 5.5);
 * weighted importance sampling is biased (its first estimate is simply the first return), but the bias
 * goes away, and its variance is far smaller: it is usually the better one.
 *
 * Discounting-aware importance sampling (Section 5.8): with γ < 1, the return G_t is a mix of flat
 * partial returns Ḡ_{t:h} = r_t + ... + r_{h-1}, without discounting, Ḡ_{t:h} weighted by (1 - γ) γ^{h-t-1}
 * and the full Ḡ_{t:T} by γ^{T-t-1}. Each partial return only needs the ratios of the actions before h:
 *
 * ordinary:  V(s) = Σ_{t in T(s)} [ (1 - γ) Σ_{h=t+1}^{T-1} γ^{h-t-1} ρ_{t:h-1} Ḡ_{t:h} + γ^{T-t-1} ρ_{t:T-1} Ḡ_{t:T} ] / |T(s)|
 * weighted:  the same sum, divided by Σ_{t in T(s)} [ (1 - γ) Σ_{h=t+1}^{T-1} γ^{h-t-1} ρ_{t:h-1} + γ^{T-t-1} ρ_{t:T-1} ]
 *
 * With γ = 1 they are ordinary and weighted importance sampling.
 *
 * Per-decision importance sampling (Section 5.9): each reward is weighted by the ratios of the actions
 * before it only, the later ones adding nothing but variance:
 *
 * V(s) = Σ_{t in T(s)} G̃_t / |T(s)|,   G̃_t = ρ_{t:t} r_t + γ ρ_{t:t+1} r_{t+1} + ... + γ^{T-1-t} ρ_{t:T-1} r_{T-1}
 *
 * The book gives no weighted version of it, nor does relearn.
 *
 * The action values q_π(s_t, a_t) are learnt the same way, with the ratios from the *next* action
 * on (a_t is given, so its own ratio is left out: ρ_{t+1:...} instead of ρ_{t:...}), and kept in the
 * `policy` given to each call; the state values are kept in the learner, `value(s)`. Both are worked out
 * from their sums, every value being the sum of its weighted returns divided by the sum of its weights
 * (or its number of visits).
 *
 * π(a_t | s_t) and b(a_t | s_t) are read for every step before any value changes. Monte Carlo needs the
 * real return, so it only learns from episodes that ended: an episode cut short is not learnt at all.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct off_policy_monte_carlo_prediction
{
    /// discount rate - you may change this as you process episodes
    value_type gamma = 0.9;
    /// π(a | s): the probability that the target policy, the one being evaluated, takes action a in state s - required
    std::function<value_type(const state_class &, const action_class &)> target;
    /// b(a | s): the probability that the behaviour policy takes action a in state s - required
    std::function<value_type(const state_class &, const action_class &)> behaviour;
    /// the importance sampling (weighted by default)
    importance_sampling sampling = importance_sampling::weighted;
    /// true for first-visit Monte Carlo, false (the default) for every-visit, as in the book's box of Section 5.6
    bool first_visit = false;

    /**
     * @brief ctor with a discount rate @param initial_gamma (0.9 by default), the target policy
     *        @param pi giving π(a | s), the behaviour policy @param b giving b(a | s),
     *        @param initial_sampling (weighted by default) and @param initial_first_visit (false by default)
     */
    off_policy_monte_carlo_prediction(value_type initial_gamma = 0.9,
                                      std::function<value_type(const state_class &, const action_class &)> pi = nullptr,
                                      std::function<value_type(const state_class &, const action_class &)> b = nullptr,
                                      importance_sampling initial_sampling = importance_sampling::weighted,
                                      bool initial_first_visit = false);

    /**
     * @brief learn from an episode played by the behaviour policy - @param policy_map (q_π) will be modified
     * @param terminated true if the episode really ended after its last link;
     *        an episode cut short is not learnt, since its return is not known
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);

    /// @return the state value V(@param s_t) learnt, 0 if the state was never visited (or all its weights were 0)
    value_type value(const state_class & s_t) const;

    /**
     * @return the weighted return of the step at @param t of an episode, with its weight
     *         (`first` = the weighted return, `second` = its weight, 1 for the ordinary samplings)
     * @param r are the rewards r_0 ... r_{T-1}, @param ratio the ratios π(a_k | s_k) / b(a_k | s_k)
     * @param first is the first action whose ratio counts: t for a state value, t + 1 for an action value
     */
    std::pair<value_type,value_type> sample(const std::vector<value_type> & r,
                                            const std::vector<value_type> & ratio,
                                            std::size_t t,
                                            std::size_t first) const;

private:
    // the sums of the weighted returns, and of the weights, of each state: [state] => (Σ returns, Σ weights)
    std::unordered_map<state_class,
                       std::pair<value_type,value_type>,
                       hasher<state_class>> state_sums_;
    // the same sums for each state/action pair
    policy<state_class,action_class,value_type> numerators_;
    policy<state_class,action_class,value_type> denominators_;
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

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
off_policy_monte_carlo_prediction<state_class,action_class,markov_chain,value_type
                                 >::off_policy_monte_carlo_prediction(value_type initial_gamma,
                                                                      std::function<value_type(const state_class &, const action_class &)> pi,
                                                                      std::function<value_type(const state_class &, const action_class &)> b,
                                                                      importance_sampling initial_sampling,
                                                                      bool initial_first_visit)
: gamma(initial_gamma),
  target(pi),
  behaviour(b),
  sampling(initial_sampling),
  first_visit(initial_first_visit)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
std::pair<value_type,value_type> off_policy_monte_carlo_prediction<state_class,action_class,markov_chain,value_type
                                                                  >::sample(const std::vector<value_type> & r,
                                                                            const std::vector<value_type> & ratio,
                                                                            std::size_t t,
                                                                            std::size_t first) const
{
    const std::size_t T = r.size();
    const bool weighted = sampling == importance_sampling::weighted ||
                          sampling == importance_sampling::discounting_aware_weighted;
    // each loop keeps ρ_{first:k}, the ratio of the actions first ... k, as it goes (1 while there is none)
    if (sampling == importance_sampling::per_decision) {
        // G̃_t = Σ_{k=t}^{T-1} γ^{k-t} ρ_{first:k} r_k: each reward weighted by the ratios before it
        value_type G = 0;
        value_type discount = 1;
        value_type rho = 1;
        for (std::size_t k = t; k < T; k++) {
            if (k >= first) {
                rho *= ratio[k];
            }
            G += discount * rho * r[k];
            discount *= gamma;
        }
        return {G, value_type(1)};
    }
    if (sampling == importance_sampling::discounting_aware_ordinary ||
        sampling == importance_sampling::discounting_aware_weighted) {
        // the flat partial returns Ḡ_{t:h}, each weighted by (1 - γ) γ^{h-t-1} ρ_{first:h-1},
        // and the full one, Ḡ_{t:T}, by γ^{T-t-1} ρ_{first:T-1}
        value_type G = 0;
        value_type W = 0;
        value_type flat = 0;
        value_type discount = 1;
        value_type rho = 1;
        for (std::size_t h = t + 1; h <= T; h++) {
            flat += r[h - 1];
            if (h - 1 >= first) {
                rho *= ratio[h - 1];
            }
            const value_type weight = (h < T ? (1 - gamma) : value_type(1)) * discount * rho;
            G += weight * flat;
            W += weight;
            discount *= gamma;
        }
        return {G, weighted ? W : value_type(1)};
    }
    // the return G_t = r_t + γ r_{t+1} + ..., weighted by ρ_{first:T-1}
    value_type G = 0;
    value_type discount = 1;
    value_type W = 1;
    for (std::size_t k = t; k < T; k++) {
        G += discount * r[k];
        discount *= gamma;
        if (k >= first) {
            W *= ratio[k];
        }
    }
    return {W * G, weighted ? W : value_type(1)};
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void off_policy_monte_carlo_prediction<state_class,action_class,markov_chain,value_type
                                      >::operator()(markov_chain episode,
                                                    policy<state_class,action_class,value_type> & policy_map,
                                                    bool terminated)
{
    // an episode cut short has no return yet: nothing to learn
    if (!terminated || episode.size() == 0) {
        return;
    }
    assert(target && "off_policy_monte_carlo_prediction needs the target policy π(a | s)");
    assert(behaviour && "off_policy_monte_carlo_prediction needs the behaviour policy b(a | s)");
    // the rewards, and the ratio π(a_t | s_t) / b(a_t | s_t) of every step, read before any value changes
    const std::size_t T = episode.size();
    std::vector<value_type> r(T);
    std::vector<value_type> ratio(T);
    for (std::size_t t = 0; t < T; t++) {
        r[t] = episode[t].state.reward();
        const value_type b = behaviour(episode[t].state, episode[t].action);
        assert(b > 0 && "the behaviour policy took an action it gives no probability to");
        ratio[t] = target(episode[t].state, episode[t].action) / b;
    }
    // first-visit: whether each step is the first visit of its state, and of its state/action pair, in the episode
    std::unordered_map<state_class,bool,hasher<state_class>> state_seen;
    visit_counter<state_class,action_class> pair_seen;
    std::vector<bool> first_state(T), first_pair(T);
    for (std::size_t t = 0; t < T; t++) {
        first_state[t] = state_seen.emplace(episode[t].state, true).second;
        first_pair[t] = pair_seen.increment(episode[t].state, episode[t].action) == 1;
    }
    // each visit: the state value of s_t, with the ratios from a_t on, and the action value of (s_t, a_t),
    // with the ratios from a_{t+1} on, each adding its weighted return and its weight to the sums
    for (std::size_t t = 0; t < T; t++) {
        const auto & step = episode[t];
        if (!first_visit || first_state[t]) {
            auto state_sample = sample(r, ratio, t, t);
            auto & sums = state_sums_[step.state];
            sums.first += state_sample.first;
            sums.second += state_sample.second;
        }
        if (first_visit && !first_pair[t]) {
            continue;
        }
        auto action_sample = sample(r, ratio, t, t + 1);
        const value_type numerator = numerators_.value(step.state, step.action) + action_sample.first;
        const value_type denominator = denominators_.value(step.state, step.action) + action_sample.second;
        numerators_.update(step.state, step.action, numerator);
        denominators_.update(step.state, step.action, denominator);
        // q_π(s_t, a_t): the sum of its weighted returns over the sum of its weights (0 while that is 0)
        policy_map.update(step.state, step.action, denominator != 0 ? numerator / denominator : value_type(0));
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type off_policy_monte_carlo_prediction<state_class,action_class,markov_chain,value_type
                                            >::value(const state_class & s_t) const
{
    auto found = state_sums_.find(s_t);
    if (found == state_sums_.end() || found->second.second == 0) {
        return value_type(0);
    }
    return found->second.first / found->second.second;
}
} // end of namespace
#endif
