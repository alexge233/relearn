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
 * @brief relearn C++ reinforcement learning library
 * @version 0.1.0
 * @date 26-8-2016
 */
namespace relearn 
{
// @brief templated equality functor
template <typename T> struct equal;
//@ @brief templated hashing functor
template <typename T> struct hash;
/// @brief combining hashes
template <class T> void hash_combine(std::size_t& seed, const T& v);

/*******************************************************************************
 * @brief a policy is a pair of a state of type S and action of type A
 * @class policy is a pair of state, action
 * @note policy is not given a value, but maps to a value inside an episode
 *
 * A policy simply denotes the choice of doing A, while in S.
 * The A decision leads to a next S, and so a tree graph is formed.
 * We use policy in order to associate a "value" to each one.
 *******************************************************************************/
template <typename S, typename A>
class policy
{
public:
    /// 
    policy(const S & state, const A & action);
    /// @brief policy equality 
    bool operator==(const policy & rhs) const;
    /// @return reference to state
    const S & state() const;
    /// @return reference to action
    const A & action() const;
private:
    const S & __state__;
    const A & __action__;
};
// @brief hash functor for policy
template <typename S, typename A> 
struct hash<policy<S, A>>
{
    size_t operator()(const policy<S,A> & arg) const;
};
// @brief equality functor
template <typename S, typename A> 
struct equal<policy<S, A>>
{
    bool operator()(const policy<S, A> & lhs, const policy<S, A> & rhs) const;
};

/*******************************************************************************
 * @brief the class which encapsulates an episode.
 * @class episode
 * @typename S defines the state s_t
 * @typename A defines the action a_t
 *
 * This class owns a `root` state which forms a tree graph.
 * The entire tree graph is contained in the episode,
 * iteration requires to traverse states and their actions, leading
 * to next states and their actions, until no more actions exist.
 * States will have a reward R, which will be used to calculate
 * the actual policy values.
 *******************************************************************************/
template <typename S, typename A>
class episode
{
public:
    /// @brief shortcuts
    using equal = equal<policy<S,A>>;
    using hash  = hash<policy<S,A>>;
    /// create episode with root state
    episode(S state);
    /// @return a reference to root state
    S & root();
    /// @brief update value for a policy
    void update(policy<S,A> & pair, float value);
    /// @return value of policy
    float value(policy<S,A> & pair);
    /// @return max/best policy for @param state, action
    float max_policy(S & state);
    /// @brief equality is based on root state and policies
    bool operator==(const episode<S,A> & arg) const;
    /// @brief constant policy iterator
    typedef typename 
    std::unordered_map<policy<S,A>, 
                       float, 
                       hash, 
                       equal>::const_iterator policy_iterator;
    /// @brief begin const policy iterator 
    policy_iterator begin() const;
    /// @brief end const policy iterator
    policy_iterator end() const;
private:
    // root state 
    S __root__;
    // episode owns policies, mapping policies to a value
    std::unordered_map<policy<S,A>, float, hash, equal> __policies__;
};

/*******************************************************************************
 * @brief breadth-first search for an episode's states-actions MDP
 * @class breadth_first
 * @note this is an iterator class, where the container is the graph/MDP
 *
 * It should return a state from the breadth of the current layer, and only
 * return a deeper layer if current breadth has been explored.
 *******************************************************************************/
template <typename S, typename A>
class breadth_first_search
{
public:
    /// @brief construct using root state
    breadth_first_search(S & root);
    /// @return next state when given @param action
    S & next() const;
    /// @return the current state
    S & current() const;
private:
    S & __state__;
    A & __action__;
};

/*******************************************************************************
 * @brief Q-Learning update algorithm updates an episode's policies
 * @note it constructs and populates an episode's policies
 * @param gamma is the discount rate
 * @param alpha is the learning rate
 *
 * q(s_t,a_t) = q(s_t,a_t) + α * (r_{t+1} + γ * max(q(s_{t+1}, a)) - q(s_t, a_t))
 *******************************************************************************/
template <typename S, typename A> 
struct q_learning
{
    void operator()(episode<S,A> & arg, float gamma, float alpha);
};

/********************************************************************************
 *                      Implementation of hashing functors
 ********************************************************************************/
template <class T>
void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/********************************************************************************
 *                      Implementation of policy class
 ********************************************************************************/
template <typename S, typename A>
policy<S,A>::policy(const S & state, const A & action)
:__state__(state), __action__(action)
{}

template <typename S, typename A>
bool policy<S,A>::operator==(const policy & rhs) const
{
    return (this->__state__ == rhs.__state__) && 
           (this->__action == rhs.__action__);
}

template <typename S, typename A>
const S & policy<S,A>::state() const
{
    return __state__;
}

template <typename S, typename A>
const A & policy<S,A>::action() const
{
    return __action__;
}

/********************************************************************************
 *                      Implementation of episode class
 ********************************************************************************/
template <typename S, typename A> 
episode<S,A>::episode(S state)
: __root__(state)
{}

template <typename S, typename A>
S & episode<S,A>::root() 
{
    return __root__;
}

/********************************************************************************
 *                      Implementation of R.L. algorithms
 ********************************************************************************/
template <typename S, typename A> 
void q_learning<S,A>::operator()(episode<S,A> & arg, float gamma, float alpha)
{
    // TODO: iterate episode's states, actions and calculate policies
    // TODO: WARNING this episode has multiple terminal states
    //          - for each forward iteration we do a breadth-first search
    //          - until we find the reward:
    //              * create a policy for each state,action pair
    //              * update policy value using next state's reward
    //              * this may have to be repeated multiple times to update the policies
    std::cout << "TODO: create / update episode's policies" << std::endl;
    // q(s_t,a_t) = q(s_t,a_t) + α * (r_{t+1} + γ * max(q(s_{t+1}, a)) - q(s_t, a_t))
}

} // end of namespace
#endif
