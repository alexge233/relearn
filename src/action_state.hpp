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
template <typename S, typename A> class state;
template <class T> struct hasher;
/**
 * \brief the descriptor class which defines an action or state
 * \class descriptor
 * \version 0.1.0
 * \date 11-9-2016
 */
template <typename T>
class trait
{
public:
    /// \brief equality
    bool operator==(const trait<T> & rhs) const;
};
/**
 * \brief an action class - servces as an example but may be used as base
 * \class action
 * \version 0.1.0
 * \date 26-8-2016
 * \note typename S is the state trait type, typename A is the action trait type
 */
template <typename S, typename A> 
class action
{
public:

    /// \brief construct using \param next state
    action(state<S,A>, trait<A> rhs); 

    /// \brief get next state - mutable state
    state<S,A> & next() const;

    /// \brief equality operator - uses `descriptor::operator==`
    bool operator==(const action<S,A> & arg) const;

    /// descriptor used for hashing
    std::size_t hash() const;

private:
    /// next state
    std::unique_ptr<state<S,A>> __next__;
    /// action descriptor
    trait<A> __tag__;
};

/// \brief hash functor for action<S,A>
template <typename S, typename A> 
struct hasher<action<S,A>>
{
    std::size_t operator()(const action<S,A> & arg) const
    {
        return arg.hash();
    } 
};

/**
 * \brief a state class
 * \class state
 * \version 0.1.0
 * \date 26-8-2016
 *
 * The state class owns *one to many* actions of type A
 * Those actions lead to next states, e.g., a tree structure
 * \note typename S is the state trait type, typename A is the action trait type
 */
template <typename S, typename A>
class state
{
public:

    /// construct with a reward (terminal state)
    state(float reward, trait<S> arg);

    /// construct with [0...n] actions
    state(std::initializer_list<action<S,A>> actions, trait<S> arg);

    /// construct with [0...n] actions and a reward (oxymoron)
    state(std::initializer_list<action<S,A>> actions, float reward, trait<S> arg);

    /// \brief add an action - unique, no duplicates
    void operator<<(action<S,A> arg);

    /// \brief state equality - uses T::operator==
    bool operator==(const state<S,A> & arg) const;

    /// \return unique hash
    std::size_t hash() const;

    // type define constant action iterator
    typedef typename
    std::unordered_set<action<S,A>, hasher<action<S,A>>>::const_iterator action_iterator;

    /// \brief begin iterating actions
    action_iterator begin() const;

    /// \brief end of actions range 
    action_iterator end() const;

    /// \return reward
    float reward() const;
    
private:
    //  unique actions - immutable set
    std::unordered_set<action<S,A>, hasher<action<S,A>>> __actions__;
    // state reward
    float __reward__ = .0f;
    // state descriptor
    trait<S> __tag__;
};

/// \brief hash functor for state<S,A>
template <typename S, typename A> 
struct hasher<state<S,A>>
{
    std::size_t operator()(const state<S,A> & arg) const
    {
        return arg.hash();
    } 
};

/********************************************************************************
 *                      Implementation of above definitions
 ********************************************************************************/

template <typename T>
bool trait<T>::operator==(const trait<T> & rhs) const
{
    return *this == rhs;
}

// TODO: implement the classes here

} // end of namespace
#endif
