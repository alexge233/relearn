#ifndef RELEARN_PLANNING_HPP
#define RELEARN_PLANNING_HPP
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
/**
 * @file relearn/planning.hpp
 * @brief planning with a model of the world (`finite_mdp`): trajectory sampling (`trajectory_sampling`),
 *        real-time dynamic programming (`real_time_dynamic_programming`), heuristic search (`heuristic_search`)
 *        and rollout algorithms (`rollout_planner`)
 */
#include "core.hpp"
#include "dynamic_programming.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace relearn
{
/**
 * @brief draw one outcome of taking @param a in @param s from the @param model, by its probability p(s', r | s, a)
 * @param prng is the random number generator
 * @return the outcome drawn (the last one, if rounding leaves the probabilities a hair short of 1)
 */
template <class state_class,
          class action_class,
          typename value_type,
          class prng_type>
typename finite_mdp<state_class,action_class,value_type>::outcome
    sample_outcome(const finite_mdp<state_class,action_class,value_type> & model,
                   const state_class & s,
                   const action_class & a,
                   prng_type & prng);

/// @brief how `trajectory_sampling` picks the state/action pairs it updates (Section 8.6)
enum class update_distribution
{
    /// every pair in turn, one after the other, sweep after sweep
    uniform,
    /// the pairs met along trajectories simulated from the start state, ε-greedy on the values
    on_policy
};

/**
 * @struct trajectory_sampling This is planning with expected updates, distributed uniformly or on-policy
 *         (Sutton & Barto, Section 8.6, Figure 8.8)
 * @brief one expected update at a time, of the pair the distribution picks
 * @param gamma is the discount rate
 * @param epsilon is ε, the exploration of the on-policy trajectories
 * @param distribution is how the pairs are picked
 *
 * Every update is an *expected* update, of one state/action pair, from the model (equation 8.1):
 *
 * Q(s,a) = Σ_{s',r} p(s',r | s,a) * (r + γ * max_a' Q(s',a'))          (max_a' Q(s',a') = 0 if s' is terminal)
 *
 * The question of Section 8.6 is which pairs to update. `uniform` goes over every pair of the model in turn, as a sweep
 * of value iteration would. `on_policy` simulates episodes from the start state with the model, ε-greedy on the values,
 * and updates each pair as it is met, starting again from the start state whenever an episode ends: it spends its
 * updates where the policy goes, and leaves alone what it never visits. On large problems with few branches, that finds
 * a good policy for the start state much sooner; given long enough, the uniform updates catch up, and pass it.
 *
 * The values are kept in the policy given, as the learners keep theirs; a pair not in it is worth 0.
 */
template <class state_class,
          class action_class,
          typename value_type = double>
struct trajectory_sampling
{
    /// the model: p(s', r | s, a)
    using model_type = finite_mdp<state_class,action_class,value_type>;
    /// discount rate
    value_type gamma = 1;
    /// ε: the exploration of the on-policy trajectories
    value_type epsilon = 0.1;
    /// how the pairs to update are picked
    update_distribution distribution = update_distribution::on_policy;
    /// actions whose values are within this of the best are equally good (rounding makes true ties differ a little)
    value_type ties = value_type(1e-9);

    /**
     * @brief ctor with a discount rate @param initial_gamma (1 by default), the distribution of the updates
     *        @param initial_distribution (on-policy by default) and @param initial_epsilon (0.1 by default)
     */
    trajectory_sampling(value_type initial_gamma = 1,
                        update_distribution initial_distribution = update_distribution::on_policy,
                        value_type initial_epsilon = 0.1);

    /**
     * @return the expected update of (@param s, @param a), Σ p(s',r | s,a) * (r + γ * max_a' Q(s',a')),
     *         from the values of @param policy_map
     */
    value_type expected_value(const model_type & model,
                              const policy<state_class,action_class,value_type> & policy_map,
                              const state_class & s,
                              const action_class & a) const;

    /**
     * @brief make @param updates expected updates - @param policy_map will be modified
     * @param model is the model of the world
     * @param start is the state every on-policy trajectory starts from
     * @param prng draws the moves and the outcomes of the trajectories
     * The uniform updates carry on from where the last call stopped, and so do the on-policy trajectories.
     */
    template <class prng_type>
    void plan(const model_type & model,
              policy<state_class,action_class,value_type> & policy_map,
              const state_class & start,
              std::size_t updates,
              prng_type & prng);

    /// @return the expected updates made so far
    std::size_t updates() const;

private:
    // the expected updates made so far
    std::size_t updates_ = 0;
    // uniform: the state and the action of the next update
    std::size_t next_state_ = 0;
    std::size_t next_action_ = 0;
    // on-policy: the state the trajectory has reached, if one is under way
    std::vector<state_class> at_;
};

/**
 * @struct real_time_dynamic_programming This is real-time dynamic programming, RTDP (Sutton & Barto, Section 8.7)
 * @brief value iteration, but only on the states met along trajectories that follow the greedy policy
 * @param gamma is the discount rate
 *
 * Each trial starts in a start state and follows the greedy policy on the values, drawing the outcome of every move
 * from the model. At every state it reaches it makes the value iteration update (equation 4.10):
 *
 * V(s) = max_a Σ_{s',r} p(s',r | s,a) * (r + γ * V(s'))
 *
 * and then takes the best of those actions (ties broken at random). The other states are never updated. For an
 * episodic task whose values start *optimistic* (no lower than the optimal ones, and 0 at the goals), where every step
 * costs something and some policy reaches a goal from every start state for sure, the values of the states that matter
 * converge to the optimal ones, and the greedy policy is optimal from every start state, without ever sweeping the whole
 * state space: in Example 8.6 RTDP needs about half the updates of value iteration, and leaves most states untouched.
 *
 * The values are kept in `values`, as in `dynamic_programming`; a state not in it is worth 0.
 */
template <class state_class,
          class action_class,
          typename value_type = double>
struct real_time_dynamic_programming
{
    /// the model: p(s', r | s, a)
    using model_type  = finite_mdp<state_class,action_class,value_type>;
    /// V: the value of each state
    using values_type = typename dynamic_programming<state_class,action_class,value_type>::values_type;
    /// discount rate
    value_type gamma = 1;
    /// actions whose values are within this of the best are equally good (rounding makes true ties differ a little)
    value_type ties = value_type(1e-9);
    /// V(s): the value of each state, 0 for a state not in it - you may set it to start from other values
    values_type values;

    /// @brief ctor with a discount rate @param initial_gamma (1 by default)
    real_time_dynamic_programming(value_type initial_gamma = 1);

    /// @return V(@param s), or 0 if it has no value yet
    value_type value(const state_class & s) const;
    /// @return q(s,a) = Σ p(s',r | s,a) * (r + γ * V(s')), for action @param a in state @param s
    value_type q_value(const model_type & model,
                       const state_class & s,
                       const action_class & a) const;
    /**
     * @return the greedy action in @param s, the best by `q_value`, ties (within `ties`) broken at random with @param prng
     * @warning @param s must have an action
     */
    template <class prng_type>
    action_class greedy(const model_type & model,
                        const state_class & s,
                        prng_type & prng) const;

    /**
     * @brief one trial from @param start, until a terminal state (one with no action) or @param max_steps steps
     * @param model is the model of the world
     * @param prng draws the outcomes of the moves, and breaks the ties of the greedy policy
     * @return the number of steps of the trial (each one an update)
     */
    template <class prng_type>
    std::size_t trial(const model_type & model,
                      const state_class & start,
                      prng_type & prng,
                      std::size_t max_steps = 100000);

    /// @return the updates made so far, one at every step of every trial
    std::size_t updates() const;

private:
    // the greedy action in s (ties broken at random with prng), and the highest q-value, its value iteration update
    template <class prng_type>
    std::pair<action_class,value_type> best(const model_type & model,
                                            const state_class & s,
                                            prng_type & prng) const;

    // the updates made so far
    std::size_t updates_ = 0;
};

/**
 * @struct heuristic_search This is heuristic search (Sutton & Barto, Section 8.9)
 * @brief look ahead from the current state, a few moves deep, through every outcome of the model, and take the action
 *        whose backed-up value is best; the values V at the leaves are the heuristic
 * @param gamma is the discount rate
 * @param depth is how many moves to look ahead, at least 1
 *
 * The value of a state looked at from `d` more moves away is backed up from below:
 *
 * V_d(s) = max_a Σ_{s',r} p(s',r | s,a) * (r + γ * V_{d-1}(s')),   V_0(s) = V(s),   and 0 at a terminal state
 *
 * and the action chosen in s is the best one at depth `depth`. With `depth` 1 this is the greedy action on V; the
 * deeper the search, the less it depends on V being right, and the more it costs: about (actions * outcomes)^depth
 * backups for every choice. Any V will do as a heuristic, and the better it is, the shallower a search finds the best
 * move; `values` is empty by default, every state then being worth 0.
 */
template <class state_class,
          class action_class,
          typename value_type = double>
struct heuristic_search
{
    /// the model: p(s', r | s, a)
    using model_type  = finite_mdp<state_class,action_class,value_type>;
    /// V: the value of each state
    using values_type = typename dynamic_programming<state_class,action_class,value_type>::values_type;
    /// discount rate
    value_type gamma = 1;
    /// how many moves to look ahead, at least 1
    unsigned int depth = 2;
    /// actions whose values are within this of the best are equally good (rounding makes true ties differ a little)
    value_type ties = value_type(1e-9);
    /// V(s): the heuristic value of each state, 0 for a state not in it
    values_type values;

    /// @brief ctor with a discount rate @param initial_gamma (1 by default) and the depth @param initial_depth (2 by default)
    heuristic_search(value_type initial_gamma = 1,
                     unsigned int initial_depth = 2);

    /// @return V(@param s), the heuristic, or 0 if it has no value
    value_type value(const state_class & s) const;
    /// @return V_d(@param s), backed up from @param d moves below
    value_type search(const model_type & model,
                      const state_class & s,
                      unsigned int d) const;
    /// @return Σ p(s',r | s,a) * (r + γ * V_{depth-1}(s')), the backed-up value of @param a in @param s
    value_type q_value(const model_type & model,
                       const state_class & s,
                       const action_class & a) const;
    /**
     * @return the action whose backed-up value is best in @param s (the first of them in A(s), on a tie within `ties`)
     * @warning @param s must have an action
     */
    action_class choose(const model_type & model,
                        const state_class & s) const;
};

/**
 * @struct rollout_planner This is a rollout algorithm (Sutton & Barto, Section 8.10)
 * @brief estimate the value of every action in the current state by simulating episodes from it, which then follow a
 *        rollout policy, and take the action whose average return is best
 * @param gamma is the discount rate
 * @param rollouts is the number of simulated episodes for each action
 * @param max_steps is the most steps of a simulated episode
 *
 * The average return of action a, followed by the rollout policy π until the episode ends, is a Monte Carlo
 * estimate of q_π(s, a); taking the best of them is one step of policy improvement on π, made only for the state the
 * agent is in, when it is there. So the agent playing with rollouts plays at least as well as π (in expectation, and
 * given enough rollouts), and often much better, without learning anything: its estimates are thrown away after each
 * move. The episodes are drawn from the model (`sample_outcome`), which is all a rollout needs.
 *
 * `rollout_policy`, if it is set, gives the action of π in a state, among its legal ones; if it is not, π picks one of
 * them at random. It is called with the generator given to `choose`, so that the whole planner draws from one stream.
 */
template <class state_class,
          class action_class,
          typename value_type = double>
struct rollout_planner
{
    /// the model: p(s', r | s, a)
    using model_type = finite_mdp<state_class,action_class,value_type>;
    /// discount rate
    value_type gamma = 1;
    /// the number of simulated episodes for each action
    unsigned int rollouts = 100;
    /// the most steps of a simulated episode
    std::size_t max_steps = 1000;
    /// optional: π, the rollout policy, given a state, its legal actions and a generator; at random if it is not set
    std::function<action_class(const state_class &, const std::vector<action_class> &, std::mt19937 &)> rollout_policy;

    /**
     * @brief ctor with a discount rate @param initial_gamma (1 by default), the rollouts of each action
     *        @param initial_rollouts (100 by default) and the rollout policy @param pi (at random if it is not set)
     */
    rollout_planner(value_type initial_gamma = 1,
                    unsigned int initial_rollouts = 100,
                    std::function<action_class(const state_class &, const std::vector<action_class> &, std::mt19937 &)> pi = nullptr);

    /**
     * @return the return of one episode simulated from @param s, taking @param a and then following π,
     *         drawn with @param prng
     */
    value_type rollout(const model_type & model,
                       const state_class & s,
                       const action_class & a,
                       std::mt19937 & prng) const;
    /// @return the average return of `rollouts` episodes from @param s that start with @param a
    value_type estimate(const model_type & model,
                        const state_class & s,
                        const action_class & a,
                        std::mt19937 & prng) const;
    /**
     * @return the action of @param s whose estimate is best (the first of them in A(s), on a tie)
     * @warning @param s must have an action
     */
    action_class choose(const model_type & model,
                        const state_class & s,
                        std::mt19937 & prng) const;
};

/********************************************************************************
 ***********               Template Implementations
 ********************************************************************************/

template <class state_class,
          class action_class,
          typename value_type,
          class prng_type>
typename finite_mdp<state_class,action_class,value_type>::outcome
    sample_outcome(const finite_mdp<state_class,action_class,value_type> & model,
                   const state_class & s,
                   const action_class & a,
                   prng_type & prng)
{
    const auto outcomes = model.outcomes(s, a);
    assert(!outcomes.empty() && "an action must have an outcome");
    std::uniform_real_distribution<double> uniform(0, 1);
    double u = uniform(prng);
    for (const auto & o : outcomes) {
        u -= static_cast<double>(o.probability);
        if (u < 0) {
            return o;
        }
    }
    return outcomes.back();
}

// a terminal state has no action: its value, and the best value of its actions, is 0
template <class state_class,
          class action_class,
          typename value_type>
value_type best_value_of(const finite_mdp<state_class,action_class,value_type> & model,
                         const policy<state_class,action_class,value_type> & policy_map,
                         const state_class & s)
{
    const auto actions = model.actions(s);
    if (actions.empty()) {
        return value_type(0);
    }
    value_type best = -std::numeric_limits<value_type>::infinity();
    for (const auto & a : actions) {
        best = std::max(best, policy_map.value(s, a));
    }
    return best;
}

template <class state_class,
          class action_class,
          typename value_type>
trajectory_sampling<state_class,action_class,value_type
                   >::trajectory_sampling(value_type initial_gamma,
                                          update_distribution initial_distribution,
                                          value_type initial_epsilon)
: gamma(initial_gamma),
  epsilon(initial_epsilon),
  distribution(initial_distribution)
{}

template <class state_class,
          class action_class,
          typename value_type>
value_type trajectory_sampling<state_class,action_class,value_type
                              >::expected_value(const model_type & model,
                                                const policy<state_class,action_class,value_type> & policy_map,
                                                const state_class & s,
                                                const action_class & a) const
{
    // Σ p(s',r | s,a) * (r + γ * max_a' Q(s',a'))
    value_type q = 0;
    for (const auto & o : model.outcomes(s, a)) {
        q += o.probability * (o.reward + gamma * best_value_of(model, policy_map, o.next));
    }
    return q;
}

template <class state_class,
          class action_class,
          typename value_type>
template <class prng_type>
void trajectory_sampling<state_class,action_class,value_type
                        >::plan(const model_type & model,
                                policy<state_class,action_class,value_type> & policy_map,
                                const state_class & start,
                                std::size_t updates,
                                prng_type & prng)
{
    for (std::size_t k = 0; k < updates; k++) {
        if (distribution == update_distribution::uniform) {
            // the next pair in turn: the next action of the state, or the first of the next state that has one
            assert(!model.states.empty() && "the model has no state");
            std::size_t tries = 0;
            while (next_action_ >= model.actions(model.states[next_state_]).size()) {
                next_action_ = 0;
                next_state_ = (next_state_ + 1) % model.states.size();
                tries++;
                assert(tries <= model.states.size() && "the model has no action at all");
            }
            // only the assert reads it
            (void)tries;
            const auto & s = model.states[next_state_];
            const auto a = model.actions(s)[next_action_];
            policy_map.update(s, a, expected_value(model, policy_map, s, a));
            next_action_++;
        }
        else {
            // on-policy: carry on from where the trajectory is, or start one; ε-greedy on the values
            if (at_.empty()) {
                at_.push_back(start);
            }
            const state_class s = at_.front();
            const auto actions = model.actions(s);
            assert(!actions.empty() && "the start state must have an action");
            std::uniform_real_distribution<double> uniform(0, 1);
            std::uniform_int_distribution<std::size_t> pick(0, actions.size() - 1);
            std::size_t chosen = pick(prng);
            if (uniform(prng) >= static_cast<double>(epsilon)) {
                // the greedy action, ties broken at random
                value_type best = -std::numeric_limits<value_type>::infinity();
                for (const auto & a : actions) {
                    best = std::max(best, policy_map.value(s, a));
                }
                std::vector<std::size_t> equal;
                for (std::size_t i = 0; i < actions.size(); i++) {
                    if (policy_map.value(s, actions[i]) >= best - ties) {
                        equal.push_back(i);
                    }
                }
                std::uniform_int_distribution<std::size_t> tie(0, equal.size() - 1);
                chosen = equal[tie(prng)];
            }
            const auto a = actions[chosen];
            policy_map.update(s, a, expected_value(model, policy_map, s, a));
            // the next state, drawn from the model; a terminal one ends the trajectory
            const auto o = sample_outcome(model, s, a, prng);
            at_.clear();
            if (!model.actions(o.next).empty()) {
                at_.push_back(o.next);
            }
        }
        updates_++;
    }
}

template <class state_class,
          class action_class,
          typename value_type>
std::size_t trajectory_sampling<state_class,action_class,value_type>::updates() const
{
    return updates_;
}

template <class state_class,
          class action_class,
          typename value_type>
real_time_dynamic_programming<state_class,action_class,value_type
                             >::real_time_dynamic_programming(value_type initial_gamma)
: gamma(initial_gamma)
{}

template <class state_class,
          class action_class,
          typename value_type>
value_type real_time_dynamic_programming<state_class,action_class,value_type
                                        >::value(const state_class & s) const
{
    auto found = values.find(s);
    return found != values.end() ? found->second : value_type(0);
}

template <class state_class,
          class action_class,
          typename value_type>
value_type real_time_dynamic_programming<state_class,action_class,value_type
                                        >::q_value(const model_type & model,
                                                   const state_class & s,
                                                   const action_class & a) const
{
    // Σ p(s',r | s,a) * (r + γ * V(s')), a terminal state being worth 0
    value_type q = 0;
    for (const auto & o : model.outcomes(s, a)) {
        const value_type next = model.actions(o.next).empty() ? value_type(0) : value(o.next);
        q += o.probability * (o.reward + gamma * next);
    }
    return q;
}

template <class state_class,
          class action_class,
          typename value_type>
template <class prng_type>
action_class real_time_dynamic_programming<state_class,action_class,value_type
                                          >::greedy(const model_type & model,
                                                    const state_class & s,
                                                    prng_type & prng) const
{
    return best(model, s, prng).first;
}

template <class state_class,
          class action_class,
          typename value_type>
template <class prng_type>
std::pair<action_class,value_type> real_time_dynamic_programming<state_class,action_class,value_type
                                                                >::best(const model_type & model,
                                                                        const state_class & s,
                                                                        prng_type & prng) const
{
    const auto actions = model.actions(s);
    assert(!actions.empty() && "a terminal state has no greedy action");
    std::vector<value_type> q(actions.size());
    value_type best = -std::numeric_limits<value_type>::infinity();
    for (std::size_t i = 0; i < actions.size(); i++) {
        q[i] = q_value(model, s, actions[i]);
        best = std::max(best, q[i]);
    }
    // the actions within `ties` of the best are equally good
    std::vector<std::size_t> equal;
    for (std::size_t i = 0; i < actions.size(); i++) {
        if (q[i] >= best - ties) {
            equal.push_back(i);
        }
    }
    std::uniform_int_distribution<std::size_t> tie(0, equal.size() - 1);
    return std::make_pair(actions[equal[tie(prng)]], best);
}

template <class state_class,
          class action_class,
          typename value_type>
template <class prng_type>
std::size_t real_time_dynamic_programming<state_class,action_class,value_type
                                         >::trial(const model_type & model,
                                                  const state_class & start,
                                                  prng_type & prng,
                                                  std::size_t max_steps)
{
    state_class s = start;
    std::size_t steps = 0;
    while (steps < max_steps && !model.actions(s).empty()) {
        // V(s) = max_a q(s, a), then the best of those actions, and the outcome drawn from the model
        const auto chosen = best(model, s, prng);
        const auto a = chosen.first;
        values[s] = chosen.second;
        updates_++;
        steps++;
        s = sample_outcome(model, s, a, prng).next;
    }
    return steps;
}

template <class state_class,
          class action_class,
          typename value_type>
std::size_t real_time_dynamic_programming<state_class,action_class,value_type>::updates() const
{
    return updates_;
}

template <class state_class,
          class action_class,
          typename value_type>
heuristic_search<state_class,action_class,value_type
                >::heuristic_search(value_type initial_gamma,
                                    unsigned int initial_depth)
: gamma(initial_gamma),
  depth(initial_depth)
{}

template <class state_class,
          class action_class,
          typename value_type>
value_type heuristic_search<state_class,action_class,value_type
                           >::value(const state_class & s) const
{
    auto found = values.find(s);
    return found != values.end() ? found->second : value_type(0);
}

template <class state_class,
          class action_class,
          typename value_type>
value_type heuristic_search<state_class,action_class,value_type
                           >::search(const model_type & model,
                                     const state_class & s,
                                     unsigned int d) const
{
    const auto actions = model.actions(s);
    // a terminal state is worth 0; at the bottom of the search, the heuristic
    if (actions.empty()) {
        return value_type(0);
    }
    if (d == 0) {
        return value(s);
    }
    // V_d(s) = max_a Σ p(s',r | s,a) * (r + γ * V_{d-1}(s'))
    value_type best = -std::numeric_limits<value_type>::infinity();
    for (const auto & a : actions) {
        value_type q = 0;
        for (const auto & o : model.outcomes(s, a)) {
            q += o.probability * (o.reward + gamma * search(model, o.next, d - 1));
        }
        best = std::max(best, q);
    }
    return best;
}

template <class state_class,
          class action_class,
          typename value_type>
value_type heuristic_search<state_class,action_class,value_type
                           >::q_value(const model_type & model,
                                      const state_class & s,
                                      const action_class & a) const
{
    assert(depth >= 1 && "heuristic_search needs a depth of at least 1");
    value_type q = 0;
    for (const auto & o : model.outcomes(s, a)) {
        q += o.probability * (o.reward + gamma * search(model, o.next, depth - 1));
    }
    return q;
}

template <class state_class,
          class action_class,
          typename value_type>
action_class heuristic_search<state_class,action_class,value_type
                             >::choose(const model_type & model,
                                       const state_class & s) const
{
    const auto actions = model.actions(s);
    assert(!actions.empty() && "a terminal state has no action to choose");
    // the first action within `ties` of the best
    std::vector<value_type> q(actions.size());
    value_type best = -std::numeric_limits<value_type>::infinity();
    for (std::size_t i = 0; i < actions.size(); i++) {
        q[i] = q_value(model, s, actions[i]);
        best = std::max(best, q[i]);
    }
    std::size_t first = 0;
    while (q[first] < best - ties) {
        first++;
    }
    return actions[first];
}

template <class state_class,
          class action_class,
          typename value_type>
rollout_planner<state_class,action_class,value_type
               >::rollout_planner(value_type initial_gamma,
                                  unsigned int initial_rollouts,
                                  std::function<action_class(const state_class &, const std::vector<action_class> &, std::mt19937 &)> pi)
: gamma(initial_gamma),
  rollouts(initial_rollouts),
  rollout_policy(pi)
{}

template <class state_class,
          class action_class,
          typename value_type>
value_type rollout_planner<state_class,action_class,value_type
                          >::rollout(const model_type & model,
                                     const state_class & s,
                                     const action_class & a,
                                     std::mt19937 & prng) const
{
    // take a, then follow π until the episode ends (or `max_steps` steps): G = r_0 + γ r_1 + γ² r_2 + ...
    auto o = sample_outcome(model, s, a, prng);
    value_type G = o.reward;
    value_type discount = gamma;
    for (std::size_t step = 1; step < max_steps; step++) {
        const auto actions = model.actions(o.next);
        if (actions.empty()) {
            break;
        }
        action_class next = actions.front();
        if (rollout_policy) {
            next = rollout_policy(o.next, actions, prng);
        }
        else {
            std::uniform_int_distribution<std::size_t> pick(0, actions.size() - 1);
            next = actions[pick(prng)];
        }
        o = sample_outcome(model, o.next, next, prng);
        G += discount * o.reward;
        discount *= gamma;
    }
    return G;
}

template <class state_class,
          class action_class,
          typename value_type>
value_type rollout_planner<state_class,action_class,value_type
                          >::estimate(const model_type & model,
                                      const state_class & s,
                                      const action_class & a,
                                      std::mt19937 & prng) const
{
    assert(rollouts >= 1 && "rollout_planner needs at least one rollout");
    value_type total = 0;
    for (unsigned int k = 0; k < rollouts; k++) {
        total += rollout(model, s, a, prng);
    }
    return total / static_cast<value_type>(rollouts);
}

template <class state_class,
          class action_class,
          typename value_type>
action_class rollout_planner<state_class,action_class,value_type
                            >::choose(const model_type & model,
                                      const state_class & s,
                                      std::mt19937 & prng) const
{
    const auto actions = model.actions(s);
    assert(!actions.empty() && "a terminal state has no action to choose");
    std::size_t best = 0;
    value_type best_q = estimate(model, s, actions[0], prng);
    for (std::size_t i = 1; i < actions.size(); i++) {
        const value_type q = estimate(model, s, actions[i], prng);
        if (q > best_q) {
            best = i;
            best_q = q;
        }
    }
    return actions[best];
}
} // end of namespace
#endif
