#ifndef RELEARN_CORE_HPP
#define RELEARN_CORE_HPP
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
 * @file relearn/core.hpp
 * @brief the building blocks: `state`, `action`, the `link` of an episode, and the `policy` that stores the learnt values
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
#include <cstddef>
#if USING_BOOST_SERIALIZATION
#include "serialize.tpl"
#endif

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

#if USING_BOOST_SERIALIZATION
#include "serialize.tpl"
#endif
} // end of namespace
#endif
