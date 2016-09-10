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
namespace relearn {
/*
 * Current implementation of action and state serves as a base/example.
 * TODO: template them so that they capture *what* they wrap around (actions and states)
 * in a manner that makes sense: 
 *      e.g., action<unsigned int> for the gridworld action
 *            state<grid>          for the gridworld state
 *
 * Of course that's not the only way to do it, another way would be to use tag descriptors
 * or dynamic polymorphism, and so on.
 */
class state;
template <class T> struct hasher;
/**
 * \brief an action class - servces as an example but may be used as base
 * \class action
 * \version 0.1.0
 * \date 26-8-2016
 */
class action
{
public:
    /// \brief construct using \param next state
    action(state arg); 

    /// \brief get next state - mutable state
    state & next() const;

    /// equality operator
    bool operator==(const action & arg) const;

    /// descriptor used for hashing
    std::size_t hash() const;

private:
    /// next state
    std::unique_ptr<state> __next__;
};

/// \brief hash functor for actions
template <> struct hasher<action>
{
    std::size_t operator()(const action & arg) const
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
 */
class state
{
public:
 
    /// construct with a reward (terminal state)
    state(float reward);   

    /// construct with [0...n] actions
    state(std::initializer_list<action> actions);

    /// construct with [0...n] actions and a reward (oxymoron)
    state(std::initializer_list<action> actions, float reward);

    /// \brief add an action - unique, no duplicates
    void operator<<(action arg);

    /// \brief state equality
    bool operator==(const state & arg) const;

    /// \return unique hash
    std::size_t hash() const;

    // type define constant action iterator
    typedef typename
    std::unordered_set<action, hasher<action>>::const_iterator action_iterator;

    /// \brief begin iterating actions
    action_iterator begin() const;

    /// \brief end of actions range 
    action_iterator end() const;

    /// \return reward
    float reward() const;
    
private:
    //  unique actions - immutable set
    std::unordered_set<action, hasher<action>> __actions__;
    // state reward
    float __reward__ = .0f;
};

/// \brief hash functor for state
template <> struct hasher<state>
{
    std::size_t operator()(const state & arg) const
    {
        return arg.hash();
    } 
};
}
#endif
