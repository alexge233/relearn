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
#include <cassert>
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
template <class state_trait>
class state
{
public:
    /// construct with a reward (terminal state)
    state(double reward, state_trait trait);
    /// @brief state equality - uses S::operator==
    bool operator==(const state<state_trait> & arg) const;
    /// @return unique hash
    std::size_t hash() const;
    /// @return reward: 0 for normal, -1 for negative, +1 for positive
    double reward() const;
    /// @return a copy of the trait
    state_trait trait() const;
private:
    // state reward
    double __reward__;
    // state descriptor (actual object/value)
    state_trait __trait__;
};

/// @brief definition of hash functor for state<S,A>
template <class state_trait> 
struct hasher<state<state_trait>>
{
    std::size_t operator()(const state<state_trait> &arg) const;
};

/**
 * @brief an action class - wrapps around your class or pdt
 * @class action
 * @version 0.1.0
 * @date 26-8-2016
 */
template <class action_trait> 
class action
{
public:
    /// @brief construct using @param next state
    action(action_trait trait); 
    /// @brief equality operator - uses `action_trait::operator==`
    bool operator==(const action<action_trait> &arg) const;
    /// hashing 
    std::size_t hash() const;
    /// return trair copy
    action_trait trait() const;
private:
    /// action descriptor - object/value wrapped
    action_trait __trait__;
};

/// @brief definition of hash functor for action<S,A>
template <class action_trait> 
struct hasher<action<action_trait>>
{
    std::size_t operator()(const action<action_trait> &arg) const;
};

/**
 * @brief we define a markov markov_chain (episode) as a deque
 * @note it could also be an std::list or std::forward_list
 * @warning it is important that order of pairs is preserved!!!
*/
template <class state_class, class action_class>
struct link
{
    std::shared_ptr<state_class> state_t;
    std::shared_ptr<action_class> action_t;
};

/**
 * @brief a markov chain (episode) is a deque of links
 */
template <class state_class, class action_class>
using markov_chain = std::deque<link<state_class,action_class>>;

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
class policy
{
public:
    /// @brief action_map correlates a state to its experienced actions/values
    typedef std::unordered_map<action_class,
                               double,
                               hasher<action_class>> action_map;
    /// @return actions experienced for this state
    action_map actions(state_class s_t);
    /// @brief update a policy value
    void update(state_class s_t, action_class a_t, double q_value);
    /// @return value of policy
    double value(state_class s_t, action_class a_t);
    /// @return max/best policy for @param state
    double best_value(state_class s_t);
    /// @return best policy for @param state - 
    /// @warning if none are found, returns nullptr
    std::unique_ptr<action_class> best_action(state_class s_t);
private:
    // internal structure mapping states => (map of actions => values)
    std::unordered_map<state_class, 
                       std::unordered_map<action_class,double,hasher<action_class>>,
                       hasher<state_class>
                       > __policies__;
};

/// @brief definition of hash functor for state<S,A>
template <class action_class> 
struct hasher<std::unordered_map<action_class,double>>
{
    std::size_t operator()(const std::unordered_map<action_class,double> &arg) const;
};

/*******************************************************************************
 * @class q_learning This is the **deterministic** Q-Learning algorithm
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
    typedef std::tuple<state_class, action_class, double> triplet;

    const double alpha;
    const double gamma;

    q_learning(double alpha, double gamma)
    : alpha(alpha), gamma(gamma)
    {}
    /// @brief the update rule of Q-learning
    triplet q_value(
                     markov_chain<state_class,action_class> &episode,
                     typename markov_chain<state_class,action_class>::iterator &step,
                     policy<state_class,action_class> &policy_map
                   );
    /// @brief do the updating for an episode
    void operator()(
                    markov_chain<state_class,action_class> episode, 
                    policy<state_class,action_class> & policy_map
                   );
};

/********************************************************************************
 ********************************************************************************
 *********************              IMPLEMENTATIONS             *****************
 ********************************************************************************
 ********************************************************************************/

/********************************************************************************
 *                      Implementation of hashing functors
 ********************************************************************************/
template <class T>
void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <class action_trait>
std::size_t hasher<action<action_trait>>::operator()(const action<action_trait> &arg) const
{
    return arg.hash();
} 

template <class state_trait>
std::size_t hasher<state<state_trait>>::operator()(const state<state_trait> &arg) const
{
    return arg.hash();
} 

template <class action_class>
std::size_t hasher<std::unordered_map<action_class,double>>::operator()(
                            const std::unordered_map<action_class,double> &arg
                                                                       ) const
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
template <class state_trait>
state<state_trait>::state(double reward, state_trait trait)
: __reward__(reward), __trait__(trait)
{}

template <class state_trait>
double state<state_trait>::reward() const
{
    return __reward__;
}

template <class state_trait>
bool state<state_trait>::operator==(const state<state_trait> & arg) const
{
    return this->__trait__ == arg.__trait__;
}

template <class state_trait>
std::size_t state<state_trait>::hash() const
{
    return std::hash<state_trait>{}(__trait__);
}

template <class state_trait>
state_trait state<state_trait>::trait() const
{
    return __trait__;
}

/********************************************************************************
 *                      Implementation of action class
 ********************************************************************************/

template <class action_trait>
action<action_trait>::action(action_trait trait)
: __trait__(trait)
{}

template <class action_trait>
std::size_t action<action_trait>::hash() const
{
   return std::hash<action_trait>{}(__trait__); 
}

template <class action_trait>
bool action<action_trait>::operator==(const action<action_trait> & arg) const
{
    return (arg.__trait__ == this->__trait__);
}

template <class action_trait>
action_trait action<action_trait>::trait() const
{
    return __trait__;
}

/********************************************************************************
 *                      Implementation of policy class
 ********************************************************************************/

template <class state_class,class action_class>
typename policy<state_class,action_class>::action_map policy<state_class,action_class>::actions(state_class s_t)
{
    return __policies__[s_t];
}

template <class state_class,class action_class>
void policy<state_class,action_class>::update(state_class s_t, action_class a_t, double q)
{
    __policies__[s_t][a_t] = q;
}

template <class state_class,class action_class>
double policy<state_class,action_class>::value(state_class s_t, action_class a_t)
{
    return __policies__[s_t][a_t];
}

template <class state_class,class action_class>
double policy<state_class,action_class>::best_value(state_class s_t)
{
    auto it = std::max_element(__policies__[s_t].begin(), __policies__[s_t].end(),
              [&](const auto &lhs, const auto &rhs) {
                    return lhs.second < rhs.second;
              });
    if (it != __policies__[s_t].end()) {
        return it->second;
    }
    return 0.;
}

template <class state_class,class action_class>
std::unique_ptr<action_class> policy<state_class,action_class>::best_action(state_class s_t)
{
    auto it = std::max_element(__policies__[s_t].begin(), __policies__[s_t].end(),
              [&](const auto &lhs, const auto &rhs) {
                  return lhs.second < rhs.second;
              });
    if (it != __policies__[s_t].end()) {
        return std::move(std::make_unique<action_class>(it->first));
    }
    return nullptr;
}

/********************************************************************************
 *                      Implementation of R.L. algorithms
 ********************************************************************************/

template <class state_class, class action_class> 
typename q_learning<state_class,action_class>::triplet q_learning<state_class,action_class>::q_value(
                                                      markov_chain<state_class,action_class> &episode, 
                                                      typename markov_chain<state_class,action_class>::iterator &step,
                                                      policy<state_class,action_class> &policy_map
                                                   ) 
{
    // q(s_t,a_t) = q(s_t,a_t) + α * (r_{t+1} + γ * max(q(s_{t+1}, a)) - q(s_t, a_t))
    if (std::distance(step, episode.end()) != 0) {
        assert(step->action_t && step->state_t);
        auto s_t  = *(step->state_t);
        auto a_t  = *(step->action_t);
        auto next = (std::next(step, 1));
        assert(next->state_t);

        double reward = next->state_t->reward();
        double q = policy_map.value(s_t, a_t);
        double q_next_max = policy_map.best_value((*next->state_t));
        double inner = reward + (gamma * q_next_max) - q;

        return std::make_tuple(s_t, a_t, (q + (alpha * inner)));
    }
    // out of bounds
    else {
        throw std::runtime_error("illegal step");
    }
}

template <class state_class, class action_class> 
void q_learning<state_class,action_class>::operator()(
                                                      markov_chain<state_class,action_class> episode, 
                                                      policy<state_class,action_class> & policy_map
                                                     )
{
    for (auto step = episode.begin(); step != episode.end() - 1; ++step) {
        auto triplet = q_value(episode, step, policy_map);
        policy_map.update(std::get<0>(triplet),
                          std::get<1>(triplet),
                          std::get<2>(triplet));
    }
}

} // end of namespace
#endif
