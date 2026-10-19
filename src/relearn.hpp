#ifndef RELEARN_HPP
#define RELEARN_HPP
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
#include <unordered_map>
#include <deque>
#include <functional>
#include <algorithm>
#include <memory>
#include <cassert>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <random>
#include <vector>
#if USING_BOOST_SERIALIZATION
#include "serialize.tpl"
#endif
/** 
 
   ________  _______   ___       _______   ________  ________  ________      
  |\   __  \|\  ___ \ |\  \     |\  ___ \ |\   __  \|\   __  \|\   ___  \
  \ \  \|\  \ \   __/|\ \  \    \ \   __/|\ \  \|\  \ \  \|\  \ \  \\ \  \
   \ \   _  _\ \  \_|/_\ \  \    \ \  \_|/_\ \   __  \ \   _  _\ \  \\ \  \
    \ \  \\  \\ \  \_|\ \ \  \____\ \  \_|\ \ \  \ \  \ \  \\  \\ \  \\ \  \
     \ \__\\ _\\ \_______\ \_______\ \_______\ \__\ \__\ \__\\ _\\ \__\\ \__\
      \|__|\|__|\|_______|\|_______|\|_______|\|__|\|__|\|__|\|__|\|__| \|__|
 

 * @brief relearn C++ reinforcement learning library
 * @version 0.1.1
 * @date 07-July-2017
 * @author Alex Giokas <alexge233@hotmail.com>
 *
 * @note: do not confuse `state_class` with `state_trait`
 * and similarly, `action_class` with `action_trait`.
 * Whereas `state_class` refers to the template parameter being used
 * in class `policy` the `state_trait` is what characterises a `state_cass`,
 * or similarly an action. Essentially, the template parameter `state_class` 
 * used later in policy, refers to template class `state` wrapper but because this
 * is a template-based library, you must roll out your own `state` class(es) provided
 * that it:
 *	- is hashable (e.g., can be used with `std::hash` or exposes a `hash()` method
 *  - is comparable (e.g., can be used with `operator<`)
 * Similary `action` takes the same approach.
 */
namespace relearn 
{
// templated equality functor
template <class T> struct equal;
// templated hashing functor
template <class T> struct hasher;
// combining hashes
template <class T> void hash_combine(std::size_t& seed, const T& v);

/**
 * @brief a state class
 * @class state
 * @version 0.1.1
 * @date 28-June-2017
 *
 * The state class is **desribed** by the template parameter `state_trait`
 * which can be virtually anything, provided you can use it to calculate a hash.
 * You are the one who decides what `state_trait` is depending on your application.
 * The ReLearn library takes a **copy** of the state_trait, so if using pointers,
 * please make sure you understand what those implications are.
 */
template <class state_trait,
          typename value_type = double>
class state
{
public:
    using trait_type = state_trait;
    /// construct without a reward (R set to zero)
    state(state_trait trait);
    /// construct with a reward (terminal state)
    state(value_type reward, state_trait trait);
    /// late reward setting
    void set_reward(value_type reward);
    /// @brief state equality - uses `state_trait::operator==`
    bool operator==(const state & arg) const;
    /// @brief comparison / osrting operator - uses `state_trait::operator<`
    bool operator<(const state & arg) const;
    /// @return unique hash
    std::size_t hash() const;
    /// @return reward: 0 for normal, -1 for negative, +1 for positive
    value_type reward() const;
    /// @return trait
    state_trait trait() const;
private:
    // state reward
    value_type reward_;
    // state descriptor (actual object/value)
    state_trait trait_;
#if USING_BOOST_SERIALIZATION
    friend boost::serialization::access;
    // @warning - template parameter `state_trait` must be serializable
    template <typename archive>
    void serialize(archive & ar, const unsigned int version);
protected:
    state() = default;
#endif
};
template <class state_trait,
          typename value_type>
struct hasher<state<state_trait,value_type>>
{
    std::size_t operator()(const state<state_trait,value_type> & arg) const;
};
#if USING_BOOST_SERIALIZATION
template <class state_class> 
struct hasher<state_serial<state_class>>
{
    std::size_t operator()(const state_serial<state_class> & arg) const;
};
#endif
/**
 * @brief an action class - wraps around your class or pdt
 * @class action
 * @version 0.1.1
 * @date 6-July-2017
 * 
 * In Reinforcement Learning literature, most often an action
 * is an affector which changes the previous state to a new/next state.
 * It must wrap around template parameter `action_trait`.
 * You have to decide what an `action` is in the context of your application,
 * how memory is managed and so on so forth.
 * Do bear in mind that class `policy` keeps a copy of the action(s).
 */
template <class action_trait> 
class action
{
public:
    using trait_type = action_trait;
    /// @brief construct using @param next state
    action(action_trait trait); 
    /// @brief equality operator - uses `action_trait::operator==`
    bool operator==(const action<action_trait> & arg) const;
    /// @brief comparison / sorting operator - uses `action_trait::operator<`
    bool operator<(const action<action_trait> & arg) const;
    /// hashing 
    std::size_t hash() const;
    /// return trait 
    action_trait trait() const;
private:
    // action descriptor - object/value wrapped
    action_trait trait_;
#if USING_BOOST_SERIALIZATION
    friend class boost::serialization::access;
    // @warning - template parameter `action_trait` must be serializable
    template <typename archive>
    void serialize(archive & ar, const unsigned int version);
protected:
    action() = default;
#endif
};
template <class action_trait> 
struct hasher<action<action_trait>>
{
    std::size_t operator()(const action<action_trait> & arg) const;
};
#if USING_BOOST_SERIALIZATION
template <class action_class> 
struct hasher<action_serial<action_class>>
{
    std::size_t operator()(const action_serial<action_class> & arg) const;
};
#endif
/**
 * @struct link
 * @brief a simple `link` or pair for joining state-actions in the MDP
 * @date 29-06-2017
 * @version 0.1.0
 */
template <class state_class, 
          class action_class>
struct link
{
    state_class state;
    action_class action;
    /// @brief sort/comparison operator
    bool operator<(const link<state_class,action_class> & arg) const;
    /// @brief equality comparisong operator
    bool operator==(const link<state_class,action_class> & arg) const;
};

/**
 * @brief the class which encapsulates learnt policies, actions and values
 * @class policy
 *
 * Template parameter `state_class` defines the state `s_t`
 * Please note that it must be hashable (@see above class `state`)
 *
 * Template parameter `action_class` defines the action `a_t`
 * Please not that it must be hashabe (@see above class `action`)
 *
 * Template parameter `value_type` defaults to a double, but feel free to change it
 *
 * This class owns all mapped state-action-policy values (it keeps copies)
 * It learns which are better than others, by observing terminal state rewards.
 * The actual **value** is not calculated in this class (it is agnostic in that
 * respect) but is instead calculated using another algorith (Q-Learning, etc.)
 *
 * Use this class in combination with an algorithm to train it (e.g., `q_learning`)
 *
 * If you need access to the underlying policies (std::unordered_map) then you
 * must inherit from this class. By doing so you accept responsibility since
 * you will have read-write access directly.
 */
template <class state_class, 
          class action_class,
          typename value_type = double>
class policy
{
public:
    /// @brief action_map correlates a state to its experienced actions/values
    using action_map = std::unordered_map<action_class,
                                          value_type,
                                          hasher<action_class>>;
    /// @return actions experienced for this state (empty if the state is unknown)
    action_map actions(const state_class & s_t) const;
    /// @brief update a policy value
    void update(const state_class & s_t, 
                const action_class & a_t, 
                value_type q_value);
    /// @return value of policy, or 0 if the state/action pair is unknown
    value_type value(const state_class & s_t, 
                     const action_class & a_t) const;
    /// @return max/best policy value for @param state, or NaN if the state is unknown
    value_type best_value(const state_class & s_t) const;
    /** 
     * @return best policy for @param state - 
     * @warning if none are found, returns nullptr
     */
    std::unique_ptr<action_class> best_action(const state_class & s_t) const;
    /**
     * @return a pair of action/value if one exists or a
     * pair of <nullptr,NaN> if one doesn't exist
     */
    std::pair<std::unique_ptr<action_class>,value_type> best(const state_class & s_t) const;
    /**
     * @brief concatenate policies, using @param arg
     * @warning the policy Q-values of @param arg take precedence `this` policies,
     * meaning that if duplicates are found, `arg`'s values will be used,
     * thereby overwritting old Q-values stored
     */
    void operator+=(const policy<state_class,action_class,value_type> & arg);
protected:
#if USING_BOOST_SERIALIZATION
    friend class boost::serialization::access;
    // serialize method
    template <typename archive>
    void serialize(archive & ar, const unsigned int version);
    // actual policies use the `_serial` wrapper from `serialize.tpl`
    using state_key  = state_serial<state_class>;
    using action_key = action_serial<action_class>;
#else
    using state_key  = state_class;
    using action_key = action_class;
#endif
    // Q-values of the actions experienced in one state
    using action_values = std::unordered_map<action_key,
                                             value_type,
                                             hasher<action_key>>;
    // policies maps is: [state][action] => Q-value
    std::unordered_map<state_key,
                       action_values,
                       hasher<state_key>
                       > policies_;
private:
    /// @return the actions stored for @param s_t, or nullptr if the state is unknown
    const action_values * find_actions(const state_class & s_t) const;
    /// @return the stored (action, value) with the highest value, or nullptr if none
    const typename action_values::value_type * best_entry(const state_class & s_t) const;
};

/*******************************************************************************
 * Schedules: values that change as learning goes on
 *
 * A schedule is called with a count k = 1, 2, 3, ... and returns a value.
 * Use one for the learning rate α (k is the number of times a state/action
 * pair has been updated, see `learning_rate` in the learners) or for the
 * exploration rate ε (k is the episode).
 *
 * For the learning rate, Q-learning, SARSA and Expected SARSA converge when,
 * for every state/action pair, the rates used on its updates satisfy
 *
 *      Σ_k α(k) = ∞   and   Σ_k α(k)² < ∞      (Sutton & Barto, Section 2.5, equation 2.7)
 *
 * which `polynomial_decay` meets for 0.5 < exponent ≤ 1.
 */

/**
 * @struct polynomial_decay
 * @brief a schedule that decays as v(k) = initial / k^exponent, for k = 1, 2, ...
 *
 * With exponent = 1 and initial = 1, v(k) = 1/k: as a learning rate, each value
 * is then the plain average of everything it has learnt from.
 * With 0.5 < exponent ≤ 1, as a learning rate, it meets the conditions above.
 */
template <typename value_type = double>
struct polynomial_decay
{
    /// the value at k = 1
    value_type initial = 1;
    /// how fast the value decays: 1 for 1/k, 0.5 ... 1 to meet the conditions above
    value_type exponent = 1;
    /// default ctor: v(k) = 1/k
    polynomial_decay() = default;
    /// ctor with the value @param initial_value at k = 1, and the decay @param decay_exponent (the `exponent`)
    polynomial_decay(value_type initial_value, 
                     value_type decay_exponent);
    /// @return the value at @param k, for k = 1, 2, ...
    value_type operator()(std::size_t k) const;
};

/**
 * @struct exponential_decay
 * @brief a schedule that decays as v(k) = initial * rate^(k-1), but not below a minimum
 *
 * Often used for the exploration rate ε. As a learning rate it does not meet the
 * conditions above: Σ v(k) stays finite, so learning stops before the values are reached
 * (unless the minimum is above 0, which keeps it learning at that rate).
 */
template <typename value_type = double>
struct exponential_decay
{
    /// the value at k = 1
    value_type initial = 1;
    /// the factor applied at each step, 0 < rate < 1
    value_type rate = 0.99;
    /// the value never falls below this
    value_type minimum = 0;
    /// default ctor: v(k) = 0.99^(k-1)
    exponential_decay() = default;
    /// ctor with the value @param initial_value at k = 1, the @param decay_rate (the `rate`)
    /// and the @param minimum_value
    exponential_decay(value_type initial_value, 
                      value_type decay_rate, 
                      value_type minimum_value);
    /// @return the value at @param k, for k = 1, 2, ...
    value_type operator()(std::size_t k) const;
};

/**
 * @struct linear_decay
 * @brief a schedule that goes in a straight line from `initial` to `last` over `steps` steps,
 *        and then stays at `last`
 *
 * Often used for the exploration rate ε: explore a lot at first, then a little.
 */
template <typename value_type = double>
struct linear_decay
{
    /// the value at k = 1
    value_type initial = 1;
    /// the value from k = steps + 1 on
    value_type last = 0;
    /// the number of steps to go from `initial` to `last`
    std::size_t steps = 1;
    /// default ctor: 1 at k = 1, then 0
    linear_decay() = default;
    /// ctor going from @param initial_value to @param last_value over @param step_count steps
    linear_decay(value_type initial_value, 
                 value_type last_value, 
                 std::size_t step_count);
    /// @return the value at @param k, for k = 1, 2, ...
    value_type operator()(std::size_t k) const;
};

/**
 * @class visit_counter
 * @brief counts how many times each state/action pair has been updated
 *
 * The learners use it to give each pair its own learning rate, α = α(n),
 * where n is the number of updates of that pair.
 */
template <class state_class, 
          class action_class>
class visit_counter
{
public:
    /// @return how many times (@param s_t, @param a_t) has been counted
    std::size_t count(const state_class & s_t, 
                      const action_class & a_t) const;
    /// @brief count one more visit of (@param s_t, @param a_t) @return its visits, this one included
    std::size_t increment(const state_class & s_t, 
                          const action_class & a_t);
    /// @brief forget every count
    void clear();
private:
    // visits of each action, per state
    using action_counts = std::unordered_map<action_class,
                                             std::size_t,
                                             hasher<action_class>>;
    std::unordered_map<state_class,
                       action_counts,
                       hasher<state_class>> counts_;
};

/**
 * @brief the learning rate of one update of (@param s_t, @param a_t)
 * @param learning_rate gives the rate of the n-th update of a pair; if it is not set, @param alpha is used
 * @param visits counts the updates of each pair: it is incremented here
 * @return `learning_rate(n)` for the n-th update of (s_t, a_t), or `alpha`
 */
template <class state_class, 
          class action_class,
          typename value_type>
value_type step_size(const std::function<value_type(std::size_t)> & learning_rate,
                     value_type alpha,
                     visit_counter<state_class,action_class> & visits,
                     const state_class & s_t,
                     const action_class & a_t);

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
                    unsigned int index,
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
 * as more episodes are observed.
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
                    unsigned int index,
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

/**
 * @struct sarsa This is the **on-policy** TD control algorithm (Sutton & Barto, Section 6.4)
 * @brief SARSA updates policy values using episodes (`markov_chain`)
 * @param gamma is the discount rate
 * @param alpha is the learning rate
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * SARSA learns from the quintuple (s_t, a_t, r_t, s_{t+1}, a_{t+1}), which gives
 * it its name. Where Q-learning uses the best action of the next state, SARSA
 * uses the action a_{t+1} that the agent actually took next:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t + γ * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))
 *
 * and for the last link, which has no next state:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t - Q(s_t, a_t))
 *
 * Because it learns the value of the actions really taken, exploration included,
 * SARSA learns the value of the policy the agent follows: it is **on-policy**.
 * So the episodes must be played with the policy being learnt (for example
 * ε-greedy on the values of `policy_map`). The extra last link of an episode cut
 * short must hold the action the agent would have taken next.
 */
template <class state_class, 
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double> 
struct sarsa
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
    sarsa(value_type initial_alpha = 0.9,
          value_type initial_gamma = 0.9);
    /// @brief the update rule of SARSA
    triplet q_value(markov_chain & episode,
                    unsigned int index,
                    policy<state_class,action_class,value_type> & policy_map);

    /**
     * @brief do the updating for an episode - @param policy_map will be modified
     * @param terminated true if the episode really ended after its last link.
     *        Pass false for an episode cut short (e.g., by a step limit): its last
     *        link is then not learnt as the end of the game, and only serves as
     *        the next state and action of the link before it.
     */
    void operator()(markov_chain episode, 
                    policy<state_class,action_class,value_type> & policy_map,
                    bool terminated = true);
};

/**
 * @struct expected_sarsa This is the Expected SARSA algorithm (Sutton & Barto, Section 6.6)
 * @brief Expected SARSA updates policy values using episodes (`markov_chain`)
 * @param gamma is the discount rate
 * @param alpha is the learning rate
 * @param probabilities is the policy π the agent follows
 *
 * Episode convention: same as `q_learning`, `s_t.reward()` is the reward r_t
 * received after taking a_t in s_t.
 *
 * Instead of the action taken next (SARSA) or the best one (Q-learning),
 * Expected SARSA uses the value the next state is *expected* to have under the
 * policy π the agent follows: the average of its action values, each weighted
 * by the probability that π takes it:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t + γ * Σ_a π(a | s_{t+1}) Q(s_{t+1}, a) - Q(s_t, a_t))
 *
 * and for the last link, which has no next state:
 *
 * Q(s_t,a_t) = Q(s_t,a_t) + α * (r_t - Q(s_t, a_t))
 *
 * This removes the randomness of choosing a_{t+1}, so it learns more steadily
 * than SARSA, and it can use larger learning rates.
 *
 * relearn does not know which actions are legal in your problem, or how your
 * agent explores, so you give it π: `probabilities(s)` returns each action the
 * agent may take in state s, with the probability that it takes it.
 * If `probabilities` is not set, the greedy policy is used, and the update is
 * the same as Q-learning's.
 */
template <class state_class, 
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double> 
struct expected_sarsa
{
    using triplet = std::tuple<state_class, 
                               action_class, 
                               value_type>;
    /// the actions the agent may take in a state, each with the probability of taking it
    using distribution = std::vector<std::pair<action_class,value_type>>;
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
    /// the policy π(a | s) the agent follows - if not set, the greedy policy is used
    std::function<distribution(const state_class &)> probabilities;
    /**
     * @brief ctor with a learning rate @param initial_alpha, a discount rate @param initial_gamma,
     *        both 0.9 by default, and the policy @param pi (`probabilities`), not set by default
     */
    expected_sarsa(value_type initial_alpha = 0.9,
                   value_type initial_gamma = 0.9,
                   std::function<distribution(const state_class &)> pi = nullptr);

    /**
     * @return the expected value of @param s_t under the policy π:
     *         Σ_a π(a | s_t) Q(s_t, a), using the values of @param policy_map
     */
    value_type expected_value(const state_class & s_t,
                              const policy<state_class,action_class,value_type> & policy_map) const;
    /// @brief the update rule of Expected SARSA
    triplet q_value(markov_chain & episode,
                    unsigned int index,
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
                    unsigned int index,
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
                    unsigned int index,
                    policy<state_class,action_class,value_type> & policy_map);
    /// @brief remember in the model what happened in the real step at @param index
    void remember(markov_chain & episode,
                  unsigned int index);
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
 ***********                            />                      
 ***********                      _,,  / >
 ***********                     "-=\~ \ > _
 ***********                        \\~___( ~
 ***********                       _|/---\\_
 ***********                       \        \
 ********************************************************************************/

template <class T>
void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <class action_trait>
std::size_t hasher<action<action_trait
				  >>::operator()(const action<action_trait> &arg) const
{
    return arg.hash();
} 

template <class state_trait,
          typename value_type>
std::size_t hasher<state<state_trait,value_type>
                  >::operator()(const state<state_trait,value_type> &arg) const
{
    return arg.hash();
} 

template <class state_trait,
          typename value_type>
state<state_trait,value_type>::state(state_trait trait)
: reward_(0), 
  trait_(trait)
{}

template <class state_trait,
          typename value_type>
state<state_trait,value_type>::state(value_type reward, 
			   state_trait trait)
: reward_(reward), 
  trait_(trait)
{}

template <class state_trait,
          typename value_type>
value_type state<state_trait,value_type>::reward() const
{
    return reward_;
}

template <class state_trait,
          typename value_type>
void state<state_trait,value_type>::set_reward(value_type reward)
{
    reward_ = reward;
}

template <class state_trait,
          typename value_type>
bool state<state_trait,value_type>::operator==(const state & arg) const
{
    return (this->trait_ == arg.trait_);
}

template <class state_trait,
          typename value_type>
bool state<state_trait,value_type>::operator<(const state & arg) const
{
    return (this->trait_ < arg.trait_);
}

template <class state_trait,
          typename value_type>
std::size_t state<state_trait,value_type>::hash() const
{
    return std::hash<state_trait>{}(trait_);
}

template <class state_trait,
          typename value_type>
state_trait state<state_trait,value_type>::trait() const
{
    return trait_;
}

#if USING_BOOST_SERIALIZATION
template <class state_trait,
          typename value_type>
template <typename archive>
void state<state_trait,value_type>::serialize(archive & ar, 
					                          const unsigned int version)
{
    ar & trait_;
    ar & reward_;
}
#endif

template <class action_trait>
action<action_trait>::action(action_trait trait)
: trait_(trait)
{}

template <class action_trait>
std::size_t action<action_trait>::hash() const
{
   return std::hash<action_trait>{}(trait_); 
}

template <class action_trait>
bool action<action_trait>::operator==(const action<action_trait> & arg) const
{
    return (this->trait_ == arg.trait_);
}

template <class action_trait>
bool action<action_trait>::operator<(const action<action_trait> & arg) const
{
    return (this->trait_ < arg.trait_);
}

template <class action_trait>
action_trait action<action_trait>::trait() const
{
    return trait_;
}

#if USING_BOOST_SERIALIZATION
template <class action_trait>
template <typename archive>
void action<action_trait>::serialize(archive & ar, 
									 const unsigned int version)
{
    ar & trait_;
}
#endif

template <class state_class, 
          class action_class>
bool link<state_class,action_class>::operator<(const link<state_class,
                                               action_class> & arg) const
{
    // lexicographic: compare states first, then actions
    if (this->state < arg.state) return true;
    if (arg.state < this->state) return false;
    return (this->action < arg.action);
}

template <class state_class, 
          class action_class>
bool link<state_class,action_class>::operator==(const link<state_class,
                                                action_class> & arg) const
{
    return (this->action == arg.action) &&
           (this->state  == arg.state);
}

template <class state_class,
          class action_class,
          typename value_type>
const typename policy<state_class,action_class,value_type>::action_values *
            policy<state_class,action_class,value_type>::find_actions(const state_class & s_t) const
{
    auto it = policies_.find(state_key(s_t));
    return it != policies_.end() ? &(it->second) : nullptr;
}

template <class state_class,
          class action_class,
          typename value_type>
const typename policy<state_class,action_class,value_type>::action_values::value_type *
            policy<state_class,action_class,value_type>::best_entry(const state_class & s_t) const
{
    const auto * stored = find_actions(s_t);
    if (!stored || stored->empty()) {
        return nullptr;
    }
    auto it = std::max_element(stored->begin(), 
                               stored->end(),
              [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });
    return &(*it);
}

template <class state_class,
          class action_class,
          typename value_type>
typename policy<state_class,action_class,value_type>::action_map 
            policy<state_class,action_class,value_type>::actions(const state_class & s_t) const
{
    action_map retval;
    if (const auto * stored = find_actions(s_t)) {
        for (const auto & item : *stored) {
            retval[static_cast<action_class>(item.first)] = item.second;
        }
    }
    return retval;
}

template <class state_class,
          class action_class,
          typename value_type>
void policy<state_class,action_class,value_type>::update(const state_class & s_t, 
                                                         const action_class & a_t, 
                                                         value_type q)
{
    policies_[state_key(s_t)][action_key(a_t)] = q;
}

template <class state_class,
          class action_class,
          typename value_type>
value_type policy<state_class,action_class,value_type>::value(const state_class & s_t, 
                                                              const action_class & a_t) const
{
    if (const auto * stored = find_actions(s_t)) {
        auto it = stored->find(action_key(a_t));
        if (it != stored->end()) {
            return it->second;
        }
    }
    return value_type(0);
}

template <class state_class,
          class action_class,
          typename value_type>
value_type policy<state_class,action_class,value_type>::best_value(const state_class & s_t) const
{
    const auto * entry = best_entry(s_t);
    return entry ? entry->second : std::numeric_limits<value_type>::quiet_NaN();
}

template <class state_class,
          class action_class,
          typename value_type>
std::unique_ptr<action_class> 
		policy<state_class,action_class,value_type>::best_action(const state_class & s_t) const
{
    const auto * entry = best_entry(s_t);
    return entry ? std::make_unique<action_class>(entry->first) : nullptr;
}

template <class state_class,
          class action_class,
          typename value_type>
std::pair<std::unique_ptr<action_class>,value_type>
    policy<state_class,action_class,value_type>::best(const state_class & s_t) const
{
    const auto * entry = best_entry(s_t);
    return entry ?
           std::make_pair(std::make_unique<action_class>(entry->first), entry->second) :
           std::make_pair(nullptr, std::numeric_limits<value_type>::quiet_NaN());
}

#if USING_BOOST_SERIALIZATION
template <class state_class,
          class action_class,
          typename value_type>
template <typename archive>
void policy<state_class,action_class,value_type>::serialize(archive & ar, 
                                                            const unsigned int version)
{
    ar & policies_;
}

template <class state_class>
std::size_t hasher<state_serial<state_class>
                  >::operator()(const state_serial<state_class> & arg) const
{
    return arg.hash();
}

template <class action_class> 
std::size_t hasher<action_serial<action_class>
                  >::operator()(const action_serial<action_class> & arg) const
{
    return arg.hash();
}
#endif

template <class state_class,
          class action_class,
          typename value_type>
void policy<state_class,action_class,value_type>::operator+=(const policy & arg)
{
#if USING_BOOST_SERIALIZATION
    for (const auto & s_t : arg.policies_) {
        for (const auto & a_t : s_t.second) {
            policies_[static_cast<state_class>(s_t.first)]
                        [static_cast<action_class>(a_t.first)] = a_t.second;
        }
    }
#else
    for (const auto & s_t : arg.policies_) {
        for (const auto & a_t : s_t.second) {
            policies_[s_t.first][a_t.first] = a_t.second;
        }
    }
#endif
}

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
sarsa<state_class,action_class,markov_chain,value_type
     >::sarsa(value_type initial_alpha,
              value_type initial_gamma)
: alpha(initial_alpha),
  gamma(initial_gamma)
{}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
expected_sarsa<state_class,action_class,markov_chain,value_type
              >::expected_sarsa(value_type initial_alpha,
                                value_type initial_gamma,
                                std::function<distribution(const state_class &)> pi)
: alpha(initial_alpha),
  gamma(initial_gamma),
  probabilities(pi)
{}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
typename q_learning<state_class,action_class,markov_chain,value_type>::triplet
            q_learning<state_class,action_class,markov_chain,value_type
					  >::q_value(markov_chain & episode, 
					   			 unsigned int index,
								 policy<state_class,action_class,value_type> & policy_map) 
{
    auto step = episode[index];
    // α: `alpha`, or the scheduled learning rate of this update of (s_t, a_t)
    auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
    if (index < episode.size() - 1)  {
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
                                      unsigned int index,
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
template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
typename sarsa<state_class,action_class,markov_chain,value_type>::triplet
            sarsa<state_class,action_class,markov_chain,value_type
                 >::q_value(markov_chain & episode, 
                            unsigned int index,
                            policy<state_class,action_class,value_type> & policy_map) 
{
    auto step = episode[index];
    // α: `alpha`, or the scheduled learning rate of this update of (s_t, a_t)
    auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
    // Q(s_t, a_t): the value we update
    auto q = policy_map.value(step.state, step.action);
    // r_t: the reward received after taking a_t in s_t
    auto r = step.state.reward();
    if (index < episode.size() - 1) {
        // the next link holds s_{t+1}, and a_{t+1}: the action the agent took next
        auto next   = episode[index + 1];
        auto q_next = policy_map.value(next.state, next.action);
        return std::make_tuple(step.state, step.action, 
                               q + rate * (r + (gamma * q_next) - q));
    }
    else {
        // last decision of the episode: no next state to bootstrap from,
        // but the reward may be stochastic, so average it with the learning rate
        return std::make_tuple(step.state, step.action, q + rate * (r - q));
    }
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
void sarsa<state_class,action_class,markov_chain,value_type
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
value_type expected_sarsa<state_class,action_class,markov_chain,value_type
                         >::expected_value(const state_class & s_t,
                                           const policy<state_class,action_class,value_type> & policy_map) const
{
    if (!probabilities) {
        // no policy given: follow the greedy one, and take the best value known
        auto best = policy_map.best_value(s_t);
        return std::isnan(best) ? value_type(0) : best;
    }
    // Σ_a π(a | s_t) Q(s_t, a): each action's value, weighted by its probability
    value_type expected = 0;
    for (const auto & item : probabilities(s_t)) {
        expected += item.second * policy_map.value(s_t, item.first);
    }
    return expected;
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
typename expected_sarsa<state_class,action_class,markov_chain,value_type>::triplet
            expected_sarsa<state_class,action_class,markov_chain,value_type
                          >::q_value(markov_chain & episode, 
                                     unsigned int index,
                                     policy<state_class,action_class,value_type> & policy_map) 
{
    auto step = episode[index];
    // α: `alpha`, or the scheduled learning rate of this update of (s_t, a_t)
    auto rate = step_size(learning_rate, alpha, visits, step.state, step.action);
    // Q(s_t, a_t): the value we update
    auto q = policy_map.value(step.state, step.action);
    // r_t: the reward received after taking a_t in s_t
    auto r = step.state.reward();
    if (index < episode.size() - 1) {
        // the expected value of s_{t+1} under the policy the agent follows
        auto next   = episode[index + 1];
        auto v_next = expected_value(next.state, policy_map);
        return std::make_tuple(step.state, step.action, 
                               q + rate * (r + (gamma * v_next) - q));
    }
    else {
        // last decision of the episode: no next state to bootstrap from,
        // but the reward may be stochastic, so average it with the learning rate
        return std::make_tuple(step.state, step.action, q + rate * (r - q));
    }
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
void expected_sarsa<state_class,action_class,markov_chain,value_type
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
template <typename value_type>
polynomial_decay<value_type>::polynomial_decay(value_type initial_value, 
                                               value_type decay_exponent)
: initial(initial_value), 
  exponent(decay_exponent)
{}

template <typename value_type>
value_type polynomial_decay<value_type>::operator()(std::size_t k) const
{
    // v(k) = initial / k^exponent, counting from k = 1
    const value_type n = static_cast<value_type>(std::max<std::size_t>(k, 1));
    return initial / std::pow(n, exponent);
}

template <typename value_type>
exponential_decay<value_type>::exponential_decay(value_type initial_value, 
                                                 value_type decay_rate, 
                                                 value_type minimum_value)
: initial(initial_value), 
  rate(decay_rate), 
  minimum(minimum_value)
{}

template <typename value_type>
value_type exponential_decay<value_type>::operator()(std::size_t k) const
{
    // v(k) = initial * rate^(k-1), but never below the minimum
    const value_type n = static_cast<value_type>(std::max<std::size_t>(k, 1) - 1);
    return std::max(minimum, initial * std::pow(rate, n));
}

template <typename value_type>
linear_decay<value_type>::linear_decay(value_type initial_value, 
                                       value_type last_value, 
                                       std::size_t step_count)
: initial(initial_value), 
  last(last_value), 
  steps(step_count)
{}

template <typename value_type>
value_type linear_decay<value_type>::operator()(std::size_t k) const
{
    // the fraction of the way from `initial` to `last`: 0 at k = 1, 1 from k = steps + 1
    const std::size_t done = std::min(std::max<std::size_t>(k, 1) - 1, steps);
    const value_type fraction = steps == 0 ? value_type(1) 
                                           : static_cast<value_type>(done) / static_cast<value_type>(steps);
    return initial + (last - initial) * fraction;
}

template <class state_class, 
          class action_class>
std::size_t visit_counter<state_class,action_class>::count(const state_class & s_t, 
                                                           const action_class & a_t) const
{
    auto state = counts_.find(s_t);
    if (state == counts_.end()) {
        return 0;
    }
    auto entry = state->second.find(a_t);
    return entry == state->second.end() ? 0 : entry->second;
}

template <class state_class, 
          class action_class>
std::size_t visit_counter<state_class,action_class>::increment(const state_class & s_t, 
                                                               const action_class & a_t)
{
    return ++counts_[s_t][a_t];
}

template <class state_class, 
          class action_class>
void visit_counter<state_class,action_class>::clear()
{
    counts_.clear();
}

template <class state_class, 
          class action_class,
          typename value_type>
value_type step_size(const std::function<value_type(std::size_t)> & learning_rate,
                     value_type alpha,
                     visit_counter<state_class,action_class> & visits,
                     const state_class & s_t,
                     const action_class & a_t)
{
    // no schedule: the same learning rate for every update
    if (!learning_rate) {
        return alpha;
    }
    // the n-th update of (s_t, a_t) uses the n-th learning rate of the schedule
    return learning_rate(visits.increment(s_t, a_t));
}
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
                                        unsigned int index,
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
    if (index < episode.size() - 1) {
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
                             unsigned int index,
                             policy<state_class,action_class,value_type> & policy_map)
{
    auto step = episode[index];
    // Q(s_t, a_t), and r_t, the reward received after a_t
    auto q = policy_map.value(step.state, step.action);
    auto r = step.state.reward();
    // the last decision of an episode has no next state to bootstrap from
    if (index == episode.size() - 1) {
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
                       unsigned int index)
{
    auto step = episode[index];
    // what happened: the reward, and the state reached (the last link of an episode ends it)
    const bool terminal = index == episode.size() - 1;
    outcome happened = {step.state.reward(),
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

#if USING_BOOST_SERIALIZATION
#include "serialize.tpl"
#endif
} // end of namespace
#endif
