#ifndef RELEARN_BANDITS_HPP
#define RELEARN_BANDITS_HPP
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
 * @file relearn/bandits.hpp
 * @brief bandit algorithms: Upper-Confidence-Bound action selection (`ucb`) and the gradient bandit (`gradient_bandit`)
 */
#include "core.hpp"
#include "schedules.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

namespace relearn
{
/**
 * @struct ucb This is Upper-Confidence-Bound action selection (Sutton & Barto, Section 2.7)
 * @brief UCB picks the action whose value could most plausibly be the highest: its estimate plus a bonus for uncertainty
 * @param c is the degree of exploration, c > 0
 *
 * A bandit is a game with one state, where every episode is a single pull of an arm (an action).
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward R_t received after
 * taking a_t in s_t, so each link of an episode is one pull.
 *
 * The value of each action is the sample average of its rewards (Section 2.4):
 *
 * N(a) = N(a) + 1
 * Q(a) = Q(a) + 1/N(a) * (R - Q(a))
 *
 * and the action taken at step t = 1, 2, ... is (equation 2.10)
 *
 * A_t = argmax_a [ Q_t(a) + c * sqrt(ln t / N_t(a)) ]
 *
 * where N_t(a) is the number of times a was taken before step t. An action never taken
 * (N_t(a) = 0) counts as maximising, and ties are broken at random.
 *
 * Each state is its own bandit: t and N_t(a) are counted in the state the action is chosen in,
 * so t = 1 + the number of pulls learnt in that state so far.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct ucb
{
    /// c: the degree of exploration - you may change this as you go
    value_type c = 2;
    /// N(s, a): the number of times each action has been learnt (pulled) in each state
    visit_counter<state_class,action_class> visits;
    /// breaks the ties between actions of equal bound - seed it to repeat a run
    std::mt19937 prng;

    /// @brief ctor with the degree of exploration @param initial_c (2 by default) and the @param prng_seed
    ucb(value_type initial_c = 2,
        unsigned int prng_seed = 0);

    /**
     * @brief the upper confidence bound Q_t(a) + c * sqrt(ln t / N_t(a)) of one action
     * @param s_t is the state (the bandit)
     * @param a_t is the action
     * @param t is the step, 1, 2, ...
     * @param policy_map holds the values Q
     * @return the bound, or infinity if @param a_t has never been taken in @param s_t
     */
    value_type bound(const state_class & s_t,
                     const action_class & a_t,
                     std::size_t t,
                     const policy<state_class,action_class,value_type> & policy_map) const;

    /**
     * @brief choose the action of highest upper confidence bound
     * @param s_t is the state (the bandit)
     * @param actions are all the actions (arms) of @param s_t
     * @param policy_map holds the values Q
     * @return the action A_t of highest bound, ties broken at random
     */
    action_class choose(const state_class & s_t,
                        const std::vector<action_class> & actions,
                        const policy<state_class,action_class,value_type> & policy_map);

    /**
     * @brief learn from the pulls of @param episode, each link one pull - @param policy_map will be modified
     *
     * Each value becomes the sample average of the rewards of its action.
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & policy_map);
};

/**
 * @struct gradient_bandit This is the gradient bandit algorithm (Sutton & Barto, Section 2.8)
 * @brief the gradient bandit learns a preference for each action, and picks actions with the softmax of them
 * @param alpha is the step size α
 * @param baseline is true to compare each reward with R̄, the average reward so far, false to compare it with 0
 *
 * Episode convention: same as `ucb`, each link of an episode is one pull, with its reward R_t in `s_t.reward()`.
 *
 * The gradient bandit learns no values. It keeps a numerical preference H(a) for each action,
 * stored in a `policy`, and takes action a with probability (equation 2.11)
 *
 * π_t(a) = exp(H_t(a)) / Σ_b exp(H_t(b))
 *
 * After taking A_t and receiving R_t, the preferences move (equation 2.12)
 *
 * H_{t+1}(A_t) = H_t(A_t) + α * (R_t - R̄_t) * (1 - π_t(A_t))
 * H_{t+1}(a)   = H_t(a)   - α * (R_t - R̄_t) * π_t(a),     for every a ≠ A_t
 *
 * so a reward above the baseline R̄_t makes A_t more likely, and one below it less likely.
 * As in the book's current text, R̄_t is the average of the rewards up to but not including
 * step t, with R̄_1 = R_1, so the first pull of a bandit changes nothing.
 * With `baseline` false, R̄_t = 0.
 *
 * Each state is its own bandit, with its own baseline R̄.
 */
template <class state_class,
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double>
struct gradient_bandit
{
    /**
     * @struct average
     * @brief the average of the rewards received in a state, and how many there were
     */
    struct average
    {
        /// the average of the rewards so far
        value_type mean = 0;
        /// the number of rewards averaged
        std::size_t count = 0;
    };

    /// α: the step size - you may change this as you go
    value_type alpha = 0.1;
    /// true to use the average reward R̄ as the baseline (the default), false for a baseline of 0
    bool baseline = true;
    /// the rewards received in each state so far, whose average is the baseline R̄
    std::unordered_map<state_class,
                       average,
                       hasher<state_class>> rewards;
    /// draws the actions from π - seed it to repeat a run
    std::mt19937 prng;

    /**
     * @brief ctor with the step size @param initial_alpha (0.1 by default), @param initial_baseline
     *        (true by default) and the @param prng_seed
     */
    gradient_bandit(value_type initial_alpha = 0.1,
                    bool initial_baseline = true,
                    unsigned int prng_seed = 0);

    /**
     * @brief the softmax of the preferences, π_t(a)
     * @param s_t is the state (the bandit)
     * @param actions are all the actions (arms) of @param s_t
     * @param preferences holds the preferences H (0 for an action not stored yet)
     * @return the probability of each of @param actions, in the same order
     */
    std::vector<value_type> probabilities(const state_class & s_t,
                                          const std::vector<action_class> & actions,
                                          const policy<state_class,action_class,value_type> & preferences) const;

    /**
     * @brief choose an action at random, with the probabilities π_t
     * @param s_t is the state (the bandit)
     * @param actions are all the actions (arms) of @param s_t
     * @param preferences holds the preferences H
     * @return the action A_t, drawn from π_t
     */
    action_class choose(const state_class & s_t,
                        const std::vector<action_class> & actions,
                        const policy<state_class,action_class,value_type> & preferences);

    /**
     * @brief learn from the pulls of @param episode, each link one pull - @param preferences will be modified
     * @param actions are all the actions (arms) of the state, since every preference moves on every pull
     */
    void operator()(markov_chain episode,
                    policy<state_class,action_class,value_type> & preferences,
                    const std::vector<action_class> & actions);
};

/********************************************************************************
 ***********               Template Implementations
 ********************************************************************************/

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
ucb<state_class,action_class,markov_chain,value_type
   >::ucb(value_type initial_c,
          unsigned int prng_seed)
: c(initial_c),
  prng(prng_seed)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
value_type ucb<state_class,action_class,markov_chain,value_type
              >::bound(const state_class & s_t,
                       const action_class & a_t,
                       std::size_t t,
                       const policy<state_class,action_class,value_type> & policy_map) const
{
    // an action never taken counts as maximising: its bound is infinite
    const auto n = visits.count(s_t, a_t);
    if (n == 0) {
        return std::numeric_limits<value_type>::infinity();
    }
    // Q_t(a) + c * sqrt(ln t / N_t(a)), equation 2.10
    const auto q = policy_map.value(s_t, a_t);
    const auto log_t = std::log(static_cast<value_type>(t));
    return q + c * std::sqrt(log_t / static_cast<value_type>(n));
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
action_class ucb<state_class,action_class,markov_chain,value_type
                >::choose(const state_class & s_t,
                          const std::vector<action_class> & actions,
                          const policy<state_class,action_class,value_type> & policy_map)
{
    assert(!actions.empty() && "ucb needs at least one action to choose from");
    // the step t = 1, 2, ...: one more than the pulls learnt so far in this state
    std::size_t t = 1;
    for (const auto & a : actions) {
        t += visits.count(s_t, a);
    }
    // the actions sharing the highest bound found so far
    std::vector<action_class> best;
    value_type best_bound = 0;
    for (const auto & a : actions) {
        auto u = bound(s_t, a, t, policy_map);
        if (best.empty() || u > best_bound) {
            // a new highest bound: it is the only best action so far
            best.assign(1, a);
            best_bound = u;
        }
        else if (u == best_bound) {
            // a tie with the highest bound
            best.push_back(a);
        }
    }
    // pick one of the best actions at random
    std::uniform_int_distribution<std::size_t> pick(0, best.size() - 1);
    return best[pick(prng)];
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void ucb<state_class,action_class,markov_chain,value_type
        >::operator()(markov_chain episode,
                      policy<state_class,action_class,value_type> & policy_map)
{
    for (std::size_t i = 0; i < episode.size(); i++) {
        const auto & pull = episode[i];
        // N(a) = N(a) + 1
        auto n = visits.increment(pull.state, pull.action);
        // Q(a) = Q(a) + 1/N(a) * (R - Q(a)), the sample average (Section 2.4)
        auto q = policy_map.value(pull.state, pull.action);
        auto r = static_cast<value_type>(pull.state.reward());
        policy_map.update(pull.state, pull.action, q + (r - q) / static_cast<value_type>(n));
    }
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
gradient_bandit<state_class,action_class,markov_chain,value_type
               >::gradient_bandit(value_type initial_alpha,
                                  bool initial_baseline,
                                  unsigned int prng_seed)
: alpha(initial_alpha),
  baseline(initial_baseline),
  prng(prng_seed)
{}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
std::vector<value_type> gradient_bandit<state_class,action_class,markov_chain,value_type
                                       >::probabilities(const state_class & s_t,
                                                        const std::vector<action_class> & actions,
                                                        const policy<state_class,action_class,value_type> & preferences) const
{
    assert(!actions.empty() && "gradient_bandit needs at least one action");
    // the preferences H_t(a), and the highest of them
    std::vector<value_type> h(actions.size());
    for (std::size_t i = 0; i < actions.size(); i++) {
        h[i] = preferences.value(s_t, actions[i]);
    }
    const value_type h_max = *std::max_element(h.begin(), h.end());
    // exp(H_t(a) - max H): the same softmax, since the factor exp(-max H) cancels out,
    // but exp never overflows, as every exponent is 0 or less
    std::vector<value_type> pi(actions.size());
    value_type sum = 0;
    for (std::size_t i = 0; i < actions.size(); i++) {
        pi[i] = std::exp(h[i] - h_max);
        sum += pi[i];
    }
    // π_t(a) = exp(H_t(a)) / Σ_b exp(H_t(b)), equation 2.11
    for (auto & p : pi) {
        p /= sum;
    }
    return pi;
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
action_class gradient_bandit<state_class,action_class,markov_chain,value_type
                            >::choose(const state_class & s_t,
                                      const std::vector<action_class> & actions,
                                      const policy<state_class,action_class,value_type> & preferences)
{
    auto pi = probabilities(s_t, actions, preferences);
    // draw u in [0, 1), and walk along the actions until their probabilities add up past u
    std::uniform_real_distribution<value_type> draw(0, 1);
    const value_type u = draw(prng);
    value_type total = 0;
    for (std::size_t i = 0; i < actions.size(); i++) {
        total += pi[i];
        if (u < total) {
            return actions[i];
        }
    }
    // the probabilities may add up to a hair under 1: u fell in that gap
    return actions.back();
}

template <class state_class,
          class action_class,
          typename markov_chain,
          typename value_type>
void gradient_bandit<state_class,action_class,markov_chain,value_type
                    >::operator()(markov_chain episode,
                                  policy<state_class,action_class,value_type> & preferences,
                                  const std::vector<action_class> & actions)
{
    for (std::size_t i = 0; i < episode.size(); i++) {
        const auto & pull = episode[i];
        const auto r = static_cast<value_type>(pull.state.reward());
        auto & received = rewards[pull.state];
        // the baseline R̄_t: the average of the rewards before this one, with R̄_1 = R_1;
        // or 0, without a baseline
        value_type r_bar = 0;
        if (baseline) {
            r_bar = received.count == 0 ? r : received.mean;
        }
        assert(std::find(actions.begin(), actions.end(), pull.action) != actions.end()
               && "the action taken must be one of the actions given");
        // π_t(a), from the preferences before any of them moves
        auto pi = probabilities(pull.state, actions, preferences);
        for (std::size_t k = 0; k < actions.size(); k++) {
            auto h = preferences.value(pull.state, actions[k]);
            if (actions[k] == pull.action) {
                // H_{t+1}(A_t) = H_t(A_t) + α * (R_t - R̄_t) * (1 - π_t(A_t)), equation 2.12
                preferences.update(pull.state, actions[k], h + alpha * (r - r_bar) * (1 - pi[k]));
            }
            else {
                // H_{t+1}(a) = H_t(a) - α * (R_t - R̄_t) * π_t(a), for every a ≠ A_t
                preferences.update(pull.state, actions[k], h - alpha * (r - r_bar) * pi[k]);
            }
        }
        // add R_t to the average, for the next step's baseline: R̄ = R̄ + 1/n * (R_t - R̄) (Section 2.4)
        received.count++;
        received.mean += (r - received.mean) / static_cast<value_type>(received.count);
    }
}
} // end of namespace
#endif
