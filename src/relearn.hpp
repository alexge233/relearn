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
#if USING_BOOST_SERIALIZATION
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unordered_map.hpp>
#endif
/**
 * @brief relearn C++ reinforcement learning library
 * @version 0.1.1
 * @date 29-06-2017
 * @author Alex Giokas <alexge233@hotmail.com>
 *
 * @NOTE: do not confuse `state_class` with `state_trait`
 * and similarly, `action_class` with `action_trait`.
 * Whereas state_class refers to the template parameter being used
 * in class `policy` the `state_trait` is what characterises a state,
 * or similarly an action. Essentially, the template parameter `state_class` 
 * used later in policy, refers to template class `state` but because this
 * is a template-based library, you can roll your own `state` class(es) provided
 * that it has: (a) a reward, (b) is hashable and (c) is comparable.
 * Similary `action` takes the same approach.
 */
namespace relearn 
{
// templated equality functor
template <class T> struct equal;
// templated hashing functor
template <class T> struct hasher;
/// @brief combining hashes
template <class T> void hash_combine(std::size_t& seed, const T& v);

/**
 * @brief a state class
 * @class state
 * @version 0.1.1
 * @date 28-06-2017
 *
 * The state class is **desribed** by the template parameter `state_trait`
 * which can be virtually anything, provided you can use it to calculate a hash
 */
template <class state_trait,
          typename value_type = double>
class state
{
public:
    /// construct without a reward (R set to zero)
    state(state_trait trait);
    /// construct with a reward (terminal state)
    state(value_type reward, state_trait trait);
    /// @brief state equality - uses `state_trait::operator==`
    bool operator==(const state<state_trait> & arg) const;
    /// @brief comparison / osrting operator - uses `state_trait::operator<`
    bool operator<(const state<state_trait> & arg) const;
    /// @return unique hash
    std::size_t hash() const;
    /// @return reward: 0 for normal, -1 for negative, +1 for positive
    value_type reward() const;
    /// @return trait
    state_trait trait() const;
#if USING_BOOST_SERIALIZATION
    /// default empty state - used by boost serialization only
    /// @warning creating an empty state on purpose will break the
    ///          learning algorithm
    state() = default;
#endif
private:
    // state reward
    value_type __reward__;
    // state descriptor (actual object/value)
    state_trait __trait__;
#if USING_BOOST_SERIALIZATION
    friend boost::serialization::access;
    // @warning - template parameter `state_trait` must be serializable
    template <typename archive>
    void serialize(archive & ar, const unsigned int version);
#endif
};

/// @brief custom hashing functor for template class state
template <class state_trait> 
struct hasher<state<state_trait>>
{
    std::size_t operator()(const state<state_trait> & arg) const;
};

/**
 * @brief an action class - wrapps around your class or pdt
 * @class action
 * @version 0.1.0
 * @date 26-8-2016
 * 
 * In Reinforcement Learning literature, most often an action
 * is an affector which changes the previous state to a new/next state.
 * It must wrap around template parameter `action_trait`.
 */
template <class action_trait> 
class action
{
public:
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
#if USING_BOOST_SERIALIZATION
    /// default empty action - used by boost serialization only
    /// @warning creating an empty action class on purpose
    ///          will break the algorithm
    action() = default;
#endif
private:
    /// action descriptor - object/value wrapped
    action_trait __trait__;
#if USING_BOOST_SERIALIZATION
    friend boost::serialization::access;
    // @warning - template parameter `action_trait` must be serializable
    template <typename archive>
    void serialize(archive & ar, const unsigned int version);
#endif
};

/// @brief custom hashing functor for template class action
template <class action_trait> 
struct hasher<action<action_trait>>
{
    std::size_t operator()(const action<action_trait> &arg) const;
};

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

/*******************************************************************************
 * @brief the class which encapsulates learnt policies, actions and values
 * @class policy
 *
 * Template parameter `state_class` defines the state s_t.
 * Please note that it must be hashable (@see above class `state`).
 *
 * Template parameter `action_class` defines the action a_t.
 * Please not that it must be hashabe (@see above class `action`).
 *
 * Template parameter `value_type` defaults to a double, but feel free to change.
 *
 * This class owns all mapped state-action-policy values.
 * It learns which are better than others, by observing terminal rewards.
 * The actual **value** is not calculated in this class (it is agnostic in that
 * respect) but is instead calculated using another algorith (Q-Learning, etc.)
 *******************************************************************************/
template <class state_class, 
          class action_class,
          typename value_type = double>
class policy
{
public:
    /// @brief action_map correlates a state to its experienced actions/values
    typedef std::unordered_map<action_class,
                               value_type,
                               hasher<action_class>> action_map;
    /// @return actions experienced for this state
    action_map actions(state_class s_t);
    /// @brief update a policy value
    void update(state_class s_t, action_class a_t, value_type q_value);
    /// @return value of policy
    value_type value(state_class s_t, action_class a_t);
    /// @return max/best policy for @param state
    value_type best_value(state_class s_t);
    /// @return best policy for @param state - 
    /// @warning if none are found, returns nullptr
    std::unique_ptr<action_class> best_action(state_class s_t);
private:
    // internal structure mapping states => (map of actions => values)
    std::unordered_map<state_class, 
                       std::unordered_map<action_class,
                                          value_type,
                                          hasher<action_class>>,
                       hasher<state_class>
                       > __policies__;
#if USING_BOOST_SERIALIZATION
    friend class boost::serialization::access;
    template <typename archive>
    void serialize(archive & ar, const unsigned int version);
#endif
};

/// @brief definition of hash functor for state<S,A>
template <class action_class,
         typename value_type>
struct hasher<std::unordered_map<action_class,value_type>>
{
    std::size_t operator()(const std::unordered_map<action_class,value_type> &arg) const;
};

/*******************************************************************************
 * @class q_learning This is the **deterministic** Q-Learning algorithm
 * @brief Q-Learning update algorithm updates an episode's policies
 * @note it constructs and populates an episode's policies
 * @param gamma is the discount rate
 * @param alpha is the learning rate
 *
 * q(s_t,a_t) = q(s_t,a_t) + α * (r_{t+1} + γ * max(q(s_{t+1}, a)) - q(s_t, a_t))
 *******************************************************************************/
template <class state_class, 
          class action_class,
          typename markov_chain = std::deque<link<state_class,action_class>>,
          typename value_type = double> 
struct q_learning
{
    typedef std::tuple<state_class, 
                       action_class, 
                       value_type> triplet;
    /// learning rate
    const double alpha;
    /// discount rate
    const double gamma;
    /// @brief the update rule of Q-learning
    triplet q_value(markov_chain &episode,
                    typename markov_chain::iterator &step,
                    policy<state_class,action_class> &policy_map);
    /// @brief do the updating for an episode
    void operator()(markov_chain episode, 
                    policy<state_class,action_class> & policy_map);
};

/********************************************************************************
 ***********               Template Implementations             *****************
 ********************************************************************************/

template <class T>
void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <class action_trait>
std::size_t hasher<action<action_trait>>::operator()(const action<action_trait> &arg) const
{
    return arg.hash();
} 

template <class state_trait>
std::size_t hasher<state<state_trait>>::operator()(const state<state_trait> &arg) const
{
    return arg.hash();
} 

template <class action_class,
          typename value_type>
std::size_t 
    hasher<std::unordered_map<action_class,value_type>>::operator()
                            (
                                const std::unordered_map<action_class,value_type> &arg
                            ) const
{
    std::size_t seed;
    for (const auto & pair : arg) {
        hash_combine(seed, pair.first.hash());
    }
    return seed;
}

template <class state_trait,
          typename value_type>
state<state_trait,value_type>::state(state_trait trait)
: __reward__(0), __trait__(trait)
{}

template <class state_trait,
          typename value_type>
state<state_trait,value_type>::state(value_type reward, state_trait trait)
: __reward__(reward), __trait__(trait)
{}

template <class state_trait,
          typename value_type>
value_type state<state_trait,value_type>::reward() const
{
    return __reward__;
}

template <class state_trait,
          typename value_type>
bool state<state_trait,value_type>::operator==(const state<state_trait> & arg) const
{
    return (this->__trait__ == arg.__trait__);
}

template <class state_trait,
          typename value_type>
bool state<state_trait,value_type>::operator<(const state<state_trait> & arg) const
{
    return (this->__trait__ < arg.__trait__);
}

template <class state_trait,
          typename value_type>
std::size_t state<state_trait,value_type>::hash() const
{
    return std::hash<state_trait>{}(__trait__);
}

template <class state_trait,
          typename value_type>
state_trait state<state_trait,value_type>::trait() const
{
    return __trait__;
}

#if USING_BOOST_SERIALIZATION
template <class state_trait,
          typename value_type>
template <typename archive>
void state<state_trait,value_type>::serialize(archive & ar, const unsigned int version)
{
    ar & __trait__;
    ar & __reward__;
}
#endif

template <class action_trait>
action<action_trait>::action(action_trait trait)
: __trait__(trait)
{}

template <class action_trait>
std::size_t action<action_trait>::hash() const
{
   return std::hash<action_trait>{}(__trait__); 
}

template <class action_trait>
bool action<action_trait>::operator==(const action<action_trait> & arg) const
{
    return (arg.__trait__ == this->__trait__);
}

template <class action_trait>
bool action<action_trait>::operator<(const action<action_trait> & arg) const
{
    return (arg.__trait__ < this->__trait__);
}

template <class action_trait>
action_trait action<action_trait>::trait() const
{
    return __trait__;
}

#if USING_BOOST_SERIALIZATION
template <class action_trait>
template <typename archive>
void action<action_trait>::serialize(archive & ar, const unsigned int version)
{
    ar & __trait__;
}
#endif

template <class state_class,
          class action_class,
          typename value_type>
typename policy<state_class,action_class,value_type>::action_map 
                policy<state_class,action_class,value_type>::actions(state_class s_t)
{
    return __policies__[s_t];
}

template <class state_class,
          class action_class,
          typename value_type>
void policy<state_class,action_class,value_type>::update(state_class s_t, 
                                                         action_class a_t, 
                                                         value_type q)
{
    __policies__[s_t][a_t] = q;
}

template <class state_class,
          class action_class,
          typename value_type>
value_type policy<state_class,action_class,value_type>::value(state_class s_t, action_class a_t)
{
    return __policies__[s_t][a_t];
}

template <class state_class,
          class action_class,
          typename value_type>
value_type policy<state_class,action_class,value_type>::best_value(state_class s_t)
{
    auto it = std::max_element(__policies__[s_t].begin(), __policies__[s_t].end(),
              [&](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });
    return it != __policies__[s_t].end() ? it->second : 0.;
}

template <class state_class,
          class action_class,
          typename value_type>
std::unique_ptr<action_class> policy<state_class,action_class,value_type>::best_action(state_class s_t)
{
    auto it = std::max_element(__policies__[s_t].begin(), __policies__[s_t].end(),
              [&](const auto &lhs, const auto &rhs) {
                  return lhs.second < rhs.second;
              });
    return it != __policies__[s_t].end() ?
           std::move(std::make_unique<action_class>(it->first)) : nullptr;
}

#if USING_BOOST_SERIALIZATION
template <class state_class,
          class action_class,
          typename value_type>
template <typename archive>
void policy<state_class,action_class,value_type>::serialize(archive & ar, const unsigned int version)
{
    ar & __policies__;
}
#endif

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
typename q_learning<state_class,action_class,markov_chain,value_type>::triplet 
                    q_learning<state_class,action_class,markov_chain,value_type>::q_value
                    (
                          markov_chain & episode, 
                          typename markov_chain::iterator & step,
                          policy<state_class,action_class> & policy_map
                    ) 
{
    assert(step);
    // q(s_t,a_t) = q(s_t,a_t) + α * (r_{t+1} + γ * max(q(s_{t+1}, a)) - q(s_t, a_t))
    if (std::distance(step, episode.end()) != 0) {
        auto next = std::next(step, 1);
        value_type q = policy_map.value(step->state, step->action);
        value_type q_next_max = policy_map.best_value((next->state));
        value_type inner = next->state.reward() + (gamma * q_next_max) - q;
        return std::make_tuple(step->state, step->action, (q + (alpha * inner)));
    }
    else {
        throw std::runtime_error("illegal step");
    }
}

template <class state_class, 
          class action_class,
          typename markov_chain,
          typename value_type>
void q_learning<state_class,action_class,markov_chain,value_type>::operator()
                (
                   markov_chain episode, 
                   policy<state_class,action_class> & policy_map
                )
{
    for (auto step = episode.begin(); step != episode.end() - 1; ++step)
    {
        auto triplet = q_value(episode, step, policy_map);
        policy_map.update(std::get<0>(triplet), 
                          std::get<1>(triplet), 
                          std::get<2>(triplet));
    }
}

template <class state_class, 
          class action_class>
bool link<state_class,action_class>::operator<(const link<state_class,
                                                          action_class> & arg) const
{
    return (this->action < arg.action) &&
           (this->state  < arg.state);
}

template <class state_class, 
          class action_class>
bool link<state_class,action_class>::operator==(const link<state_class,
                                                           action_class> & arg) const
{
    return (this->action == arg.action) &&
           (this->state  == arg.state);
}
} // end of namespace
#endif
