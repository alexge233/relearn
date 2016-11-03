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
#include <deque>
#include <functional>
#include <algorithm>
#include <memory>
#include <math.h>
/**
 * @brief relearn C++ reinforcement learning library
 * @version 0.1.0
 * @date 26-8-2016
 * @author Alex Giokas <alexge233@hotmail.com>
 *
 * @NOTE: do not confuse `state_class` with `state_trait`
 * and similarly, `action_class` with `action_trait`
 */
namespace relearn 
{
// @brief templated equality functor
template <class T> struct equal;
//@ @brief templated hashing functor
template <class T> struct hasher;
/// @brief combining hashes
template <class T> void hash_combine(std::size_t& seed, const T& v);

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

    /// construct with a reward (terminal state)
    state(float reward, state_trait trait);
    
    /// @brief state equality - uses S::operator==
    bool operator==(const state_t & arg) const;
    
    /// @return unique hash
    std::size_t hash() const;
    
    /// @return reward: 0 for normal, -1 for negative, +1 for positive
    float reward() const;

    /// @return a copy of the trait
    state_trait trait() const;

private:
    // state reward
    float __reward__ = 0;
    // state descriptor (actual object/value)
    state_trait __trait__;
};

/// @brief definition of hash functor for state<S,A>
template <class state_trait, class action_trait> 
struct hasher<state<state_trait,action_trait>>
{
    std::size_t operator()(const state<state_trait,action_trait> &arg) const;
};

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
    
    /// @brief get next state - mutable state
    state_class & next() const;
    
    /// @brief equality operator - uses `action_trait::operator==`
    bool operator==(const action_class &arg) const;

    /// hashing 
    std::size_t hash() const;

    /// return trair copy
    action_trait trait() const;

private:
    /// next state - action owns it (forward declaration)
    state_class & __next__;
    /// action descriptor - object/value wrapped
    action_trait __trait__;
};

/// @brief definition of hash functor for action<S,A>
template <class state_trait, class action_trait> 
struct hasher<action<state_trait,action_trait>>
{
    std::size_t operator()(const action<state_trait,action_trait> &arg) const;
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
    typedef std::tuple<state_class,action_class,float> policy;
    typedef std::unordered_map<action_class,float,hasher<action_class>> action_row;

    /// @return a reference to root state
    state_class root();

    /// @brief equality is based on root state and policies
    bool operator==(const episode<state_class,action_class> &arg) const;

    void set_state(const state_class s_t);

    /// @brief add or update a state
    void set_state(const state_class s_t, const action_class a_t);

    /// @return actions experienced for this state
    action_row actions(const state_class &arg);

    /// @brief update a policy value
    void operator<<(const policy &arg);

    /// @return value of policy
    float policy_value(
                       state_class s_t,
                       action_class a_t
                      );
    
    /// @return max/best policy for @param state
    float best_policy_value(const state_class &arg);

    /// @return best policy for @param state
    std::unique_ptr<action_class> best_policy(const state_class &arg);

    typename std::deque<state_class>::iterator begin();

    typename std::deque<state_class>::iterator end();

    std::size_t size() const;

private:
    std::deque<state_class> __states__;

    // episode owns policies, mapping states => (map of actions => values)
    std::unordered_map<state_class, 
                       std::unordered_map<action_class,float,hasher<action_class>>,
                       hasher<state_class>
                       > __policies__;
};

/// @brief definition of hash functor for state<S,A>
template <class action_class> 
struct hasher<std::unordered_map<action_class,float>>
{
    std::size_t operator()(const std::unordered_map<action_class,float> &arg) const;
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
    float q_value(episode<state_class,action_class> & arg,
                  const state_class & s_t, 
                  const action_class & a_t, 
                  float alpha,
                  float gamma);

    /// @brief do the updating for an episode
    void operator()(episode<state_class,action_class> &arg, float alpha, float gamma);
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

template <class action_class>
std::size_t hasher<std::unordered_map<action_class,float>>::operator()(const std::unordered_map<action_class,float> &arg) const
{
    std::size_t seed;
    for (const auto & pair : arg) {
        hash_combine(seed, pair.first.hash());
    }
    return seed;
}

/********************************************************************************
 *                      Implementation of state class
 ********************************************************************************/

template <class state_trait, class action_trait>
state<state_trait, action_trait>::state(float reward, state_trait trait)
: __reward__(reward), __trait__(trait)
{}

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
state<state_trait,action_trait>& action<state_trait, action_trait>::next() const
{
    return __next__;
}

template <class state_trait, class action_trait>
action_trait action<state_trait,action_trait>::trait() const
{
    return __trait__;
}


/********************************************************************************
 *                      Implementation of episode class
 ********************************************************************************/

template <class state_class, class action_class>
state_class episode<state_class, action_class>::root()
{
    return __states__.front();
}

template <class state_class, class action_class>
void episode<state_class,action_class>::set_state(const state_class s_t)
{
    if (__policies__.find(s_t) == __policies__.end()) {
        auto res = __policies__.insert({s_t, action_row{}}); 
        if (res.second) {
            __states__.push_back(s_t);
        }
    }
    else {
        __policies__[s_t] = action_row{};
    }
}

template <class state_class, class action_class>
void episode<state_class,action_class>::set_state(
                                                   const state_class s_t, 
                                                   const action_class a_t
                                                 )
{
    if (__policies__.find(s_t) == __policies__.end()) {
        action_row row = {{a_t, 0.f}};
        auto res = __policies__.insert({s_t, row}); 
        if (res.second) {
            __states__.push_back(s_t);
        }
    }
    else {
        __policies__[s_t] = action_row{{a_t, 0.f}};
    }
}

template <class state_class, class action_class>
class episode<state_class,action_class>::action_row 
                            episode<state_class,action_class>::actions(const state_class &s_t)
{
    return __policies__[s_t];
}

template <class state_class, class action_class>
void episode<state_class,action_class>::operator<<(const std::tuple<state_class,action_class,float> &arg)
{
    __policies__[std::get<0>(arg)][std::get<1>(arg)] = std::get<2>(arg);
}

template <class state_class, class action_class>
float episode<state_class,action_class>::policy_value(
                                                        state_class s_t,
                                                        action_class a_t
                                                      )
{
    return __policies__[s_t][a_t];
}

template <class state_class, class action_class>
float episode<state_class,action_class>::best_policy_value(const state_class &s_t)
{
    auto it = std::max_element(__policies__[s_t].begin(), __policies__[s_t].end(),
              [&](const auto &lhs, const auto &rhs) {
                    return lhs.second < rhs.second;
              });
    if (it != __policies__[s_t].end()) {
        return it->second;
    }
    else {
        return 0.f;
    }
}

template <class state_class, class action_class>
std::unique_ptr<action_class> episode<state_class,action_class>::best_policy(const state_class &arg)
{
    auto it = std::max_element(__policies__[arg].begin(), __policies__[arg].end(),
              [&](const auto &lhs, const auto &rhs) {
                  return lhs.second < rhs.second;
              });
    if (it != __policies__[arg].end()) {
        return std::move(std::make_unique<action_class>(it->first));
    }
    else {
        return nullptr;
    }
}

template <class state_class, class action_class>
typename std::deque<state_class>::iterator episode<state_class,action_class>::begin()
{
    return __states__.begin(); 
}

template <class state_class, class action_class>
typename std::deque<state_class>::iterator episode<state_class,action_class>::end()
{
    return __states__.end(); 
}

template <class state_class, class action_class>
std::size_t episode<state_class,action_class>::size() const
{
    return __states__.size();
}

/********************************************************************************
 *                      Implementation of R.L. algorithms
 ********************************************************************************/
template <class state_class, class action_class> 
float q_learning<state_class,action_class>::q_value(
                                                    episode<state_class,action_class> &arg, 
                                                    const state_class & s_t, 
                                                    const action_class & a_t, 
                                                    float alpha,
                                                    float gamma
                                                   )
{
    float reward = a_t.next().reward();
    if (reward == 1.f) {
        return reward;
    }
    if (reward == -1.f) {
        return reward;
    }
    else {
        float q_old = arg.policy_value(s_t, a_t);
        float q_next_max = arg.best_policy_value(a_t.next());
        std::cout << "\t\tq(t) = " << q_old << std::endl
                  << "\t\tα = " << alpha << std::endl
                  << "\t\tR = " << reward << std::endl
                  << "\t\tγ = " << gamma << std::endl
                  << "\t\tq(t+1) = " << q_next_max << std::endl;
        float inner = gamma * q_next_max;
        inner = reward + inner;
        inner = q_old + inner;
        float q = q_old + alpha * inner;
        std::cout << "\tq(t) = " << q << std::endl;
        return q;
    }
}

template <class state_class, class action_class> 
void q_learning<state_class,action_class>::operator()(
                                                      episode<state_class,action_class> &arg, 
                                                      float alpha, 
                                                      float gamma
                                                     )
{
    for (const auto &s_t : arg) {
        const auto action_matrix = arg.actions(s_t);
        for (const auto &action_row : action_matrix) {
            arg << std::make_tuple(s_t, 
                                   action_row.first, 
                                   q_value(arg, s_t, action_row.first, alpha, gamma));
        }
    }
}

} // end of namespace
#endif
