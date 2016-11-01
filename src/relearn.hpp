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
#include <functional>
#include <algorithm>
/**
 * @brief relearn C++ reinforcement learning library
 * @version 0.1.0
 * @date 26-8-2016
 * @author Alex Giokas <alexge233@hotmail.com>
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
struct hash<policy<S,A>>
{
    size_t operator()(const policy<S,A> & arg) const;
};
// @brief equality functor
template <typename S, typename A> 
struct equal<policy<S,A>>
{
    bool operator()(const policy<S,A> & lhs, const policy<S, A> & rhs) const;
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
    typedef std::pair<policy<S,A>, float> policy_value;

    /// create episode with root state
    episode(S state);
    
    /// @return a reference to root state
    S & root() const;
    
    /// @return value of policy - @warning will return zero if not found
    float value(const policy<S,A> & arg) const;
    
    /// @return max/best policy for @param state, action
    float max_policy(const S &arg) const;
    
    /// @brief equality is based on root state and policies
    bool operator==(const episode<S,A> & arg) const;

    /// @brief insert a policy and set its value
    bool operator<<(policy_value arg);

    /// @brief update a policy by setting its value
    void operator<<=(policy_value arg);

    /// TODO: implement an iterator for the policies

private:
    using equals = equal<policy<S,A>>;
    using hashes = hash<policy<S,A>>;
    // root state 
    S __root__;
    // episode owns policies, mapping policies to a value - no order or sorting
    std::unordered_map<policy<S,A>, float, hashes, equals> __policies__;
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
    /// @brief the update rule of Q-learning
    float q_value(const episode<S,A> & arg,
                  const S & s_t, 
                  const A & a_t, 
                  float alpha,
                  float gamma);

    /// @brief do the updating for an episode
    void operator()(episode<S,A> & arg, float alpha, float gamma);
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

template <typename S, typename A> 
bool equal<policy<S,A>>::operator()(const policy<S,A> &lhs, const policy<S,A> &rhs) const
{
    return (lhs.state() == rhs.state()) && (lhs.action() == rhs.action());
}

template <typename S, typename A> 
std::size_t hash<policy<S,A>>::operator()(const policy<S,A> & arg) const
{
    std::size_t seed;
    hash_combine(seed, arg.state().hash());
    hash_combine(seed, arg.action().hash());
    return seed;
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

template <typename S, typename A>
float episode<S,A>::value(const policy<S,A> & arg) const
{
    auto it = __policies__.find(arg);
    return (it != __policies__.end() ? it->second : 0.f);
}

template <typename S, typename A>
float episode<S,A>::max_policy(const S & arg) const
{
    std::vector<float> filter;
    std::for_each(__policies__.begin(), __policies__.end(), [&](const auto rhs) {
        if (rhs.first.state() == arg) {
            filter.push_back(rhs.second);
        }
    });
    return *(std::max_element(filter.begin(), filter.end()));
}

template <typename S, typename A>
bool episode<S,A>::operator<<(policy_value arg)
{
    return __policies__.insert({std::get<0>(arg), std::get<1>(arg)}).second;
}

template <typename S, typename A>
void episode<S,A>::operator<<=(policy_value arg)
{
    __policies__[std::get<0>(arg)] = std::get<1>(arg);
}

/********************************************************************************
 *                      Implementation of R.L. algorithms
 ********************************************************************************/
template <typename S, typename A> 
float q_learning<S,A>::q_value(
                                const episode<S,A> & arg, 
                                const S & s_t, 
                                const A & a_t, 
                                float alpha,
                                float gamma
                               )
{
    float q_old = arg.value(policy<S,A>(s_t, a_t));
    float reward = a_t.next().reward();
    float q_next_max = arg.max_policy(a_t.next());
    return q_old + alpha * (reward + (gamma * q_next_max) + q_old);
}

template <typename S, typename A> 
void q_learning<S,A>::operator()(episode<S,A> & arg, float alpha, float gamma)
{
    std::function<void(const S &s_t)> func = [&](const S &s_t) {
        for (const A & a_t : s_t) {
            if (!(arg << std::make_pair(policy<S,A>(s_t, a_t), 
                                        q_value(arg, s_t, a_t, alpha, gamma))))
            {
                arg <<= std::make_pair(policy<S,A>(s_t, a_t), 
                                       q_value(arg, s_t, a_t, alpha, gamma));
            }
            func(a_t.next());
        }
    };
    func(arg.root());
}

} // end of namespace
#endif
