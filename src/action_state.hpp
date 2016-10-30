#ifndef RELEARN_ACTION_STATE_HPP
#define RELEARN_ACTION_STATE_HPP
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
namespace relearn
{
// forward declare state
template <typename state_trait, typename action_trait> class state;
template <class T> struct hasher;
/**
 * @brief an action class - servces as an example but may be used as base
 * @class action
 * @version 0.1.0
 * @date 26-8-2016
 * @note typename S is the state trait type, typename A is the action trait type
 */
template <typename state_trait, typename action_trait> 
class action
{
public:
    using state_t = state<state_trait, action_trait>;
    using action_t = action<state_trait, action_trait>;
    /// @brief construct using @param next state
    action(state_t state, action_trait trait); 
    /// @brief get next state - mutable state
    state_t & next() const;
    /// @brief equality operator - uses `descriptor::operator==`
    bool operator==(const action_t & arg) const;
    /// descriptor used for hashing
    std::size_t hash() const;
private:
    /// next state
    state_t __next__;
    /// action descriptor
    action_trait __trait__;
};

/// @brief hash functor for action<S,A>
template <typename state_trait, typename action_trait> 
struct hasher<action<state_trait, action_trait>>
{
    std::size_t operator()(const action<state_trait, action_trait> & arg) const;
};

/**
 * @brief a state class
 * @class state
 * @version 0.1.0
 * @date 26-8-2016
 *
 * The state class owns *one to many* actions of type A
 * Those actions lead to next states, e.g., a tree structure
 * @note typename S is the state trait type, typename A is the action trait type
 */
template <typename state_trait, typename action_trait>
class state
{
public:
    using state_t = state<state_trait, action_trait>;
    using action_t = action<state_trait, action_trait>;
    /// construct with a reward (terminal state)
    state(float reward, state_trait trait);
    /// construct with [0...n] actions
    state(std::initializer_list<action_t> actions, state_trait trait);
    /// construct with [0...n] actions and a reward (oxymoron)
    state(std::initializer_list<action_t> actions, float reward, state_trait trait);
    /// @brief add an action - unique, no duplicates
    void operator<<(action_t arg);
    /// @brief state equality - uses T::operator==
    bool operator==(const state_t & arg) const;
    /// @return unique hash
    std::size_t hash() const;
    // type define constant action iterator
    typedef typename
    std::unordered_set<action_t, hasher<action_t>>::const_iterator action_iterator;
    /// @brief begin iterating actions
    action_iterator begin() const;
    /// @brief end of actions range 
    action_iterator end() const;
    /// @return reward: 0 for normal, -1 for negative, +1 for positive
    float reward() const;
private:
    //  unique actions - immutable set
    std::unordered_set<action_t, hasher<action_t>> __actions__;
    // state reward
    float __reward__ = .0f;
    // state descriptor (actual object/value)
    state_trait __trait__;
};

/// @brief hash functor for state<S,A>
template <typename state_trait, typename action_trait> 
struct hasher<state<state_trait, action_trait>>
{
    std::size_t operator()(const state<state_trait, action_trait> & arg) const;
};

/********************************************************************************
 *                      Implementation of hasher specialisations
 ********************************************************************************/
template <typename state_trait, typename action_trait>
std::size_t hasher<action<state_trait, action_trait>>::operator()(const action<state_trait, action_trait> & arg) const
{
    return arg.hash();
} 

template <typename state_trait, typename action_trait>
std::size_t hasher<state<state_trait, action_trait>>::operator()(const state<state_trait, action_trait> & arg) const
{
    return arg.hash();
} 

/********************************************************************************
 *                      Implementation of action class
 ********************************************************************************/
template <typename state_trait, typename action_trait>
action<state_trait, action_trait>::action(state<state_trait, action_trait> s, action_trait trait)
: __next__(s), __trait__(trait)
{}

template <typename state_trait, typename action_trait>
std::size_t action<state_trait, action_trait>::hash() const
{
   return std::hash<action_trait>{}(__trait__); 
}

template <typename state_trait, typename action_trait>
bool action<state_trait, action_trait>::operator==(const action_t & arg) const
{
    return (arg.__trait__ == this->__trait__);
}

template <typename state_trait, typename action_trait>
state<state_trait, action_trait>& action<state_trait, action_trait>::next() const
{
    return __next__;
}

/********************************************************************************
 *                      Implementation of state class
 ********************************************************************************/
template <typename state_trait, typename action_trait>
state<state_trait, action_trait>::state(float reward, state_trait trait)
: __reward__(reward), __trait__(trait)
{}

template <typename state_trait, typename action_trait>
state<state_trait, action_trait>::state(std::initializer_list<action_t> actions, state_trait trait)
:__reward__(0.0), __trait__(trait)
{
    __actions__.insert(__actions__.end(), actions.begin(), actions.end());
}

template <typename state_trait, typename action_trait>
state<state_trait, action_trait>::state(std::initializer_list<action_t> actions, float reward, state_trait trait)
:__reward__(reward), __trait__(trait)
{
    __actions__.insert(__actions__.end(), actions.begin(), actions.end());
}

template <typename state_trait, typename action_trait>
void state<state_trait, action_trait>::operator<<(action_t arg)
{
    __actions__.insert(arg);
}

// TODO: implement the classes here

} // end of namespace
#endif
