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
 * \class policy
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

private:
    const S & _state_;
    const A & _action_;
};


/// \brief a struct hash
template <class T> struct hash;

/// \brief combining hashes
template <class T> void hash_combine(std::size_t& seed, const T& v);

/// \brief hash functor for policy
/// \warning requires that state S, and action A are hashable
template <typename S, typename A> struct hash<policy<S,A>>
{
    std::size_t operator()(policy<S,A> const & arg) const; 
};

/**
 * \brief the class which encapsulates an episode.
 * \class episode
 * \typename S defines the state s_t
 * \typename A defines the action a_t
 * \typename N defines the value type of policy (float, double, etc)
 */
template <typename S, typename A, typename N>
class episode
{
public:
    /// empty default constructor
    episode() = default;

    /// create episode with root state
    episode(const S state);

    /// \return a reference to root state
    const S & root() const;

private:
    /// root state - immutable
    const S & _root_;
    /// episode owns policies, mapping policies to a value
    std::unordered_map<policy<S,A>, N> _policies_;
};

// TODO: updater (iterate policies updating them using Q-Learning or R-Learning)

}

/**
 * Implementation of above definitions
 */
template <class T>
void relearn::hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

#endif
