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
/**
 * \brief relearn C++ reinforcement learning library
 * \version 0.1.0
 * \date 26-8-2016
 */
namespace relearn 
{
/**
 * \brief a policy is a pair of a state of type S and action of type A
 * \class policy is a pair of state,action
 *
 * A policy simply denotes the choice of doing A, while in S.
 * The A decision leads to a next S, and so a tree graph is formed.
 * We use policy in order to associate a "value" to each one.
 */
template <typename S, typename A>
class policy
{
public:
    /// 
    policy(const S & state, const A & action);

    /// \brief policy equality 
    bool operator==(const policy & rhs) const;

    /// \return reference to state
    const S & state() const;

    /// \return reference to action
    const A & action() const;

protected:
    const S & _state_;
    const A & _action_;
};


/// \brief a struct hash - use to hash classes
template <class T> struct hash;

/// \brief combining hashes
template <class T> void hash_combine(std::size_t& seed, const T& v);

/// \brief hash functor for policy
template <typename S, typename A> struct hash<policy<S,A>>
{
    std::size_t operator()(policy<S,A> const & arg) const; 
};

/**
 * \brief the class which encapsulates an episode.
 * \class episode
 * \typename S defines the state s_t
 * \typename A defines the action a_t
 *
 * This class owns a `root` state which forms a tree graph.
 * The entire tree graph is contained in the episode,
 * iteration requires to traverse states and their actions, leading
 * to next states and their actions, until no more actions exist.
 * States will have a reward R, which will be used to calculate
 * the actual policy values.
 */
template <typename S, typename A>
class episode
{
public:

    /// empty default constructor
    episode() = default;

    /// create episode with root state
    episode(const S state);

    /// \return a reference to root state
    const S & root() const;

    /// \brief update value for a policy
    void update(policy<S,A> & pair, float value);

    /// \return value of policy
    float value(policy<S,A> & pair);

    /// \brief equality is based on root state and policies
    bool operator==(const episode<S,A> & arg) const;

    /// \brief typedefine the const policy iterator
    typedef typename std::unordered_map<policy<S,A>,float>::const_iterator const_iterator;

    /// \brief begin const policy iterator 
    const_iterator begin() const;

    /// \brief end const policy iterator
    const_iterator end() const;
    
protected:
    /// root state - immutable
    const S & _root_;
    /// episode owns policies, mapping policies to a value
    std::unordered_map<policy<S,A>, float> _policies_;
};

// TODO: updater (iterate policies updating them using Q-Learning or R-Learning)

}

/********************************************************************************
 *                      Implementation of above definitions
 ********************************************************************************/
template <class T>
void relearn::hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// TODO: impelment all other classes and code

#endif
