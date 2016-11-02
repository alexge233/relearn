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
#include <deque>
/**
 * @brief relearn C++ reinforcement learning library
 * @version 0.1.0
 * @date 26-8-2016
 * @author Alex Giokas <alexge233@hotmail.com>
 *
 * @NOTE: do not confuse `state_class` with `state_trait`
 * and similarly, `action_class` with `action_trait`
 *
 * TODO: 1. remove class policy.
 *       2. replace with: std::unordered_map<action_class, float>
 *       3. then in episode, use: std::unordered_map<state_class, std::unordered_map<action_class, float>>
 *          this is a 2D unordered_map, 1st dimension is the state, 2nd dimension is the actions
 */
namespace relearn 
{
// @brief templated equality functor
template <class T> struct equal;
//@ @brief templated hashing functor
template <class T> struct hash;
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
template <class state_class, class action_class>
class policy
{
public:
    /// 
    policy(const state_class state, const action_class action);

    /// @brief policy equality 
    bool operator==(const policy & rhs) const;

    /// @return reference to state
    state_class state() const;

    /// @return reference to action
    action_class action() const;

private:
    state_class __state__;
    action_class  __action__;
};
// @brief hash functor for policy
template <class state_class, class action_class> 
struct hash<policy<state_class,action_class>>
{
    size_t operator()(const policy<state_class,action_class> & arg) const;
};
// @brief equality functor
template <class state_class, class action_class> 
struct equal<policy<state_class,action_class>>
{
    bool operator()(const policy<state_class,action_class> & lhs, 
                    const policy<state_class,action_class> & rhs) const;
};

/*******************************************************************************
 * @brief the class which encapsulates an episode.
 * @class episode
 * @class S defines the state s_t
 * @class A defines the action a_t
 *
 * This class owns a `root` state which forms a tree graph.
 * The entire tree graph is contained in the episode,
 * iteration requires to traverse states and their actions, leading
 * to next states and their actions, until no more actions exist.
 * States will have a reward R, which will be used to calculate
 * the actual policy values.
 *******************************************************************************/
template <class state_class, class action_class>
class episode
{
public:
    typedef std::pair<policy<state_class, action_class>, float> policy_value;
    using equals = equal<policy<state_class, action_class>>;
    using hashes = hash<policy<state_class, action_class>>;

    /// create episode with root state
    episode(state_class state);
    
    /// @return a reference to root state
    state_class & root();
    
    /// @return value of policy - @warning will return zero if not found
    float value(const policy<state_class, action_class> & arg) const;
    
    /// @return max/best policy for @param state
    float best_value_policy(const state_class &arg) const;

    /// @return best policy for @param state
    policy<state_class,action_class> best_policy(const state_class &arg) const;
    
    /// @brief equality is based on root state and policies
    bool operator==(const episode<state_class, action_class> & arg) const;

    /// @brief insert a policy and set its value
    bool operator<<(policy_value arg);

    /// @brief update a policy by setting its value
    void operator<<=(policy_value arg);

private:
    // root state 
    state_class __root__;
    // episode owns policies, mapping policies to a value - no order or sorting
    std::unordered_map<policy<state_class, action_class>, 
                       float, hashes, equals> __policies__;
};

/*******************************************************************************
 * @brief Q-Learning update algorithm updates an episode's policies
 * @note it constructs and populates an episode's policies
 * @param gamma is the discount rate
 * @param alpha is the learning rate
 *
 * q(s_t,a_t) = q(s_t,a_t) + α * (r_{t+1} + γ * max(q(s_{t+1}, a)) - q(s_t, a_t))
 *******************************************************************************/
template <class state_class, class action_class> 
struct q_learning
{
    /// @brief the update rule of Q-learning
    float q_value(const episode<state_class,action_class> & arg,
                  const state_class & s_t, 
                  const action_class & a_t, 
                  float alpha,
                  float gamma);

    /// @brief do the updating for an episode
    void operator()(episode<state_class,action_class> & arg, float alpha, float gamma);
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
template <class state_class, class action_class>
policy<state_class,action_class>::policy(const state_class state, 
                                         const action_class action)
:__state__(state), __action__(action)
{}

template <class state_class, class action_class>
bool policy<state_class,action_class>::operator==(const policy & rhs) const
{
    return (this->__state__ == rhs.__state__) && 
           (this->__action == rhs.__action__);
}

template <class state_class, class action_class>
state_class policy<state_class,action_class>::state() const
{
    return __state__;
}

template <class state_class, class action_class>
action_class policy<state_class,action_class>::action() const
{
    return __action__;
}

template <class state_class, class action_class> 
bool equal<policy<state_class,action_class>>::operator()(
                                                         const policy<state_class,action_class> &lhs,
                                                         const policy<state_class,action_class> &rhs
                                                        ) const
{
    return (lhs.state() == rhs.state()) && (lhs.action() == rhs.action());
}

template <class state_class, class action_class> 
std::size_t hash<policy<state_class,action_class>>::operator()(const policy<state_class,action_class> & arg) const
{
    std::size_t seed;
    hash_combine(seed, arg.state().hash());
    hash_combine(seed, arg.action().hash());
    return seed;
}

/********************************************************************************
 *                      Implementation of episode class
 ********************************************************************************/
template <class state_class, class action_class> 
episode<state_class, action_class>::episode(state_class state)
: __root__(state)
{}

template <class state_class, class action_class>
state_class & episode<state_class, action_class>::root()
{
    return __root__;
}

template <class state_class, class action_class>
float episode<state_class, action_class>::value(const policy<state_class, action_class> & arg) const
{
    auto it = __policies__.find(arg);
    return (it != __policies__.end() ? it->second : 0.f);
}

template <class state_class, class action_class>
policy<state_class,action_class> episode<state_class,action_class>::best_policy(const state_class &arg) const
{
    std::deque<std::pair<policy<state_class,action_class>, float>> filter;
    std::for_each(__policies__.begin(), __policies__.end(), 
    [&](const auto rhs) {
        if (rhs.first.state() == arg) {
            filter.push_back(std::make_pair(rhs.first, rhs.second));
        }
    });
    return std::move((std::max_element(filter.begin(), filter.end(),
                     [](const auto & lhs, const auto & rhs) {
                        return lhs.second < rhs.second;
                     }))->first);
}

template <class state_class, class action_class>
float episode<state_class, action_class>::best_value_policy(const state_class &arg) const
{
    std::deque<float> filter;
    std::for_each(__policies__.begin(), __policies__.end(), 
    [&](const auto rhs) {
        if (rhs.first.state() == arg) {
            filter.push_back(rhs.second);
        }
    });
    return *(std::max_element(filter.begin(), filter.end()));
}

template <class state_class, class action_class>
bool episode<state_class, action_class>::operator<<(policy_value arg)
{
    return __policies__.insert({std::get<0>(arg), std::get<1>(arg)}).second;
}

template <class state_class, class action_class>
void episode<state_class, action_class>::operator<<=(policy_value arg)
{
    __policies__[std::get<0>(arg)] = std::get<1>(arg);
}

/********************************************************************************
 *                      Implementation of R.L. algorithms
 ********************************************************************************/
template <class state_class, class action_class> 
float q_learning<state_class,action_class>::q_value(
                                                    const episode<state_class, action_class> &arg, 
                                                    const state_class & s_t, 
                                                    const action_class & a_t, 
                                                    float alpha,
                                                    float gamma
                                                   )
{
    float q_old = arg.value(policy<state_class,action_class>(s_t, a_t));
    float reward = a_t.next().reward();
    float q_next_max = arg.best_value_policy(a_t.next());
    return q_old + alpha * (reward + (gamma * q_next_max) + q_old);
}

template <class state_class, class action_class> 
void q_learning<state_class,action_class>::operator()(
                                                      episode<state_class,action_class> &arg, 
                                                      float alpha, 
                                                      float gamma
                                                     )
{
    std::function<void(const state_class &s_t)> func = 
    [&](const state_class &s_t) {
        for (const action_class & a_t : s_t) {
            if (!(arg << std::make_pair(policy<state_class,action_class>(s_t, a_t), 
                                        q_value(arg, s_t, a_t, alpha, gamma))))
            {
                arg <<= std::make_pair(policy<state_class,action_class>(s_t, a_t), 
                                       q_value(arg, s_t, a_t, alpha, gamma));
            }
            func(a_t.next());
        }
    };
    func(arg.root());
}

} // end of namespace
#endif
