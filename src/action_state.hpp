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
#include <memory>
/**
 * @note class `state_trait` is the type/pdt/class used for as state
 * @note class `action_trait` is the type/pdt/class used for as action
 * @note `action_trait` and `state_trait` must be **hashable**
 */
namespace relearn
{
// forward declare state
template <class state_trait, class action_trait> 
class state;

template <class T> 
struct hasher;
/**
 * @brief an action class - wrapps around your class or pdt
 * @class action
 * @version 0.1.0
 * @date 26-8-2016
 */
template <class state_trait, class action_trait> 
class action
{
public:
    using action_class = action<state_trait,action_trait>;
    using state_class  = state<state_trait,action_trait>;

    /// @brief construct using @param next state
    action(
            state<state_trait,action_trait> state_next, 
            action_trait trait
          ); 
    
    /// @briec copy constructor
    action(const action_class &arg);

    /// @brief get next state - mutable state
    state_class next() const;
    
    /// @brief equality operator - uses `action_trait::operator==`
    bool operator==(const action_class &arg) const;

    /// @brief assignment operator
    action_class& operator=(const action_class &arg);
    
    /// descriptor used for hashing
    std::size_t hash() const;

private:
    /// next state - action owns it (forward declaration)
    state_class & __next__;
    /// action descriptor - object/value wrapped
    action_trait __trait__;
};

/// @brief definition of hash functor for action<S,A>
template <class state_trait, class action_trait> 
struct hasher<action<state_trait, action_trait>>
{
    std::size_t operator()(const action<state_trait,action_trait> &arg) const;
};

/**
 * @brief a state class
 * @class state
 * @version 0.1.0
 * @date 26-8-2016
 *
 * The state class owns *one to many* actions of type A
 * Those actions lead to next states, e.g., a tree structure
 * @note class S is the state trait type, class A is the action trait type
 */
template <class state_trait, class action_trait>
class state
{
public:
    using state_t = state<state_trait, action_trait>;
    using action_t = action<state_trait, action_trait>;

    /// construct with a reward (terminal state)
    state(float reward, state_trait trait);
    
    /// @brief add an action - unique, no duplicates
    void operator<<(action_t arg);
    
    /// @brief state equality - uses S::operator==
    bool operator==(const state_t & arg) const;
    
    /// @return unique hash
    std::size_t hash() const;
    
    // type define constant action iterator
    typedef class
    std::unordered_set<action_t, hasher<action_t>>::const_iterator 
                                                    action_iterator;
    
    /// @brief begin iterating actions
    action_iterator begin() const;
    
    /// @brief end of actions range 
    action_iterator end() const;
    
    /// @return reward: 0 for normal, -1 for negative, +1 for positive
    float reward() const;

    /// @return a copy of the trait
    state_trait trait() const;

private:
    //  unique actions - immutable set
    std::unordered_set<action_t, hasher<action_t>> __actions__;
    // state reward
    float __reward__ = .0f;
    // state descriptor (actual object/value)
    state_trait __trait__;
};

/// @brief definition of hash functor for state<S,A>
template <class state_trait, class action_trait> 
struct hasher<state<state_trait, action_trait>>
{
    std::size_t operator()(const state<state_trait, action_trait> & arg) const;
};

/********************************************************************************
 *                      Implementation of hasher functors
 ********************************************************************************/
template <class state_trait, class action_trait>
std::size_t hasher<action<state_trait,action_trait>>::operator()(const action<state_trait,action_trait> &arg) const
{
    return arg.hash();
} 

template <class state_trait, class action_trait>
std::size_t hasher<state<state_trait,action_trait>>::operator()(const state<state_trait,action_trait> &arg) const
{
    return arg.hash();
} 

/********************************************************************************
 *                      Implementation of action class
 ********************************************************************************/
template <class state_trait, class action_trait>
action<state_trait, action_trait>::action(
                                          state<state_trait, action_trait> state_next, 
                                          action_trait trait
                                         )
: __next__(state_next), __trait__(trait)
{}

template <class state_trait, class action_trait>
action<state_trait, action_trait>::action(const action<state_trait,action_trait> &arg)
: __next__(arg.__next__), __trait__(arg.__trait__) 
{}

template <class state_trait, class action_trait>
std::size_t action<state_trait, action_trait>::hash() const
{
   return std::hash<action_trait>{}(__trait__); 
}

template <class state_trait, class action_trait>
bool action<state_trait, action_trait>::operator==(const action<state_trait, action_trait> & arg) const
{
    return (arg.__trait__ == this->__trait__);
}

template <class state_trait, class action_trait>
state<state_trait,action_trait> action<state_trait, action_trait>::next() const
{
    return __next__;
}

template <class state_trait, class action_trait>
action<state_trait,action_trait>& action<state_trait,action_trait>::operator=(const action<state_trait,action_trait> &arg)
{
    if (&arg == this) {
        return *this;
    }
    this->__trait__ = arg.__trait__;
    this->__next__  = arg.__next__;
    return *this;
}

/********************************************************************************
 *                      Implementation of state class
 ********************************************************************************/

template <class state_trait, class action_trait>
state<state_trait, action_trait>::state(float reward, state_trait trait)
: __reward__(reward), __trait__(trait)
{}

template <class state_trait, class action_trait>
void state<state_trait, action_trait>::operator<<(action_t arg)
{
    __actions__.insert(arg);
}

template <class state_trait, class action_trait>
class state<state_trait,action_trait>::action_iterator state<state_trait,action_trait>::begin() const
{
    return __actions__.begin(); 
}

template <class state_trait, class action_trait>
class state<state_trait,action_trait>::action_iterator state<state_trait,action_trait>::end() const
{
    return __actions__.end(); 
}

template <class state_trait, class action_trait>
float state<state_trait, action_trait>::reward() const
{
    return __reward__;
}

template <class state_trait, class action_trait>
bool state<state_trait, action_trait>::operator==(const state_t & arg) const
{
    return this->__trait__ == arg.__trait__;
}

template <class state_trait, class action_trait>
std::size_t state<state_trait, action_trait>::hash() const
{
   return std::hash<state_trait>{}(__trait__);
}

template <class state_trait, class action_trait>
state_trait state<state_trait, action_trait>::trait() const
{
   return __trait__;
}


} // end of namespace
#endif
