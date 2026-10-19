#ifndef RELEARN_SCHEDULES_HPP
#define RELEARN_SCHEDULES_HPP
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
 * @file relearn/schedules.hpp
 * @brief schedules for the learning rate and the exploration rate, and the per-pair update counter
 */
#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>

namespace relearn
{
/*******************************************************************************
 * Schedules: values that change as learning goes on
 *
 * A schedule is called with a count k = 1, 2, 3, ... and returns a value.
 * Use one for the learning rate α (k is the number of times a state/action
 * pair has been updated, see `learning_rate` in the learners) or for the
 * exploration rate ε (k is the episode).
 *
 * For the learning rate, Q-learning, SARSA and Expected SARSA converge when,
 * for every state/action pair, the rates used on its updates satisfy
 *
 *      Σ_k α(k) = ∞   and   Σ_k α(k)² < ∞      (Sutton & Barto, Section 2.5, equation 2.7)
 *
 * which `polynomial_decay` meets for 0.5 < exponent ≤ 1.
 */

/**
 * @struct polynomial_decay
 * @brief a schedule that decays as v(k) = initial / k^exponent, for k = 1, 2, ...
 *
 * With exponent = 1 and initial = 1, v(k) = 1/k: as a learning rate, each value
 * is then the plain average of everything it has learnt from.
 * With 0.5 < exponent ≤ 1, as a learning rate, it meets the conditions above.
 */
template <typename value_type = double>
struct polynomial_decay
{
    /// the value at k = 1
    value_type initial = 1;
    /// how fast the value decays: 1 for 1/k, 0.5 ... 1 to meet the conditions above
    value_type exponent = 1;
    /// default ctor: v(k) = 1/k
    polynomial_decay() = default;
    /// ctor with the value @param initial_value at k = 1, and the decay @param decay_exponent (the `exponent`)
    polynomial_decay(value_type initial_value, 
                     value_type decay_exponent);
    /// @return the value at @param k, for k = 1, 2, ...
    value_type operator()(std::size_t k) const;
};

/**
 * @struct exponential_decay
 * @brief a schedule that decays as v(k) = initial * rate^(k-1), but not below a minimum
 *
 * Often used for the exploration rate ε. As a learning rate it does not meet the
 * conditions above: Σ v(k) stays finite, so learning stops before the values are reached
 * (unless the minimum is above 0, which keeps it learning at that rate).
 */
template <typename value_type = double>
struct exponential_decay
{
    /// the value at k = 1
    value_type initial = 1;
    /// the factor applied at each step, 0 < rate < 1
    value_type rate = 0.99;
    /// the value never falls below this
    value_type minimum = 0;
    /// default ctor: v(k) = 0.99^(k-1)
    exponential_decay() = default;
    /// ctor with the value @param initial_value at k = 1, the @param decay_rate (the `rate`)
    /// and the @param minimum_value
    exponential_decay(value_type initial_value, 
                      value_type decay_rate, 
                      value_type minimum_value);
    /// @return the value at @param k, for k = 1, 2, ...
    value_type operator()(std::size_t k) const;
};

/**
 * @struct linear_decay
 * @brief a schedule that goes in a straight line from `initial` to `last` over `steps` steps,
 *        and then stays at `last`
 *
 * Often used for the exploration rate ε: explore a lot at first, then a little.
 */
template <typename value_type = double>
struct linear_decay
{
    /// the value at k = 1
    value_type initial = 1;
    /// the value from k = steps + 1 on
    value_type last = 0;
    /// the number of steps to go from `initial` to `last`
    std::size_t steps = 1;
    /// default ctor: 1 at k = 1, then 0
    linear_decay() = default;
    /// ctor going from @param initial_value to @param last_value over @param step_count steps
    linear_decay(value_type initial_value, 
                 value_type last_value, 
                 std::size_t step_count);
    /// @return the value at @param k, for k = 1, 2, ...
    value_type operator()(std::size_t k) const;
};

/**
 * @class visit_counter
 * @brief counts how many times each state/action pair has been updated
 *
 * The learners use it to give each pair its own learning rate, α = α(n),
 * where n is the number of updates of that pair.
 */
template <class state_class, 
          class action_class>
class visit_counter
{
public:
    /// @return how many times (@param s_t, @param a_t) has been counted
    std::size_t count(const state_class & s_t, 
                      const action_class & a_t) const;
    /// @brief count one more visit of (@param s_t, @param a_t) @return its visits, this one included
    std::size_t increment(const state_class & s_t, 
                          const action_class & a_t);
    /// @brief forget every count
    void clear();
private:
    // visits of each action, per state
    using action_counts = std::unordered_map<action_class,
                                             std::size_t,
                                             hasher<action_class>>;
    std::unordered_map<state_class,
                       action_counts,
                       hasher<state_class>> counts_;
};

/**
 * @brief the learning rate of one update of (@param s_t, @param a_t)
 * @param learning_rate gives the rate of the n-th update of a pair; if it is not set, @param alpha is used
 * @param visits counts the updates of each pair: it is incremented here
 * @return `learning_rate(n)` for the n-th update of (s_t, a_t), or `alpha`
 */
template <class state_class, 
          class action_class,
          typename value_type>
value_type step_size(const std::function<value_type(std::size_t)> & learning_rate,
                     value_type alpha,
                     visit_counter<state_class,action_class> & visits,
                     const state_class & s_t,
                     const action_class & a_t);

/********************************************************************************
 ***********               Template Implementations
 ********************************************************************************/

template <typename value_type>
polynomial_decay<value_type>::polynomial_decay(value_type initial_value, 
                                               value_type decay_exponent)
: initial(initial_value), 
  exponent(decay_exponent)
{}

template <typename value_type>
value_type polynomial_decay<value_type>::operator()(std::size_t k) const
{
    // v(k) = initial / k^exponent, counting from k = 1
    const value_type n = static_cast<value_type>(std::max<std::size_t>(k, 1));
    return initial / std::pow(n, exponent);
}

template <typename value_type>
exponential_decay<value_type>::exponential_decay(value_type initial_value, 
                                                 value_type decay_rate, 
                                                 value_type minimum_value)
: initial(initial_value), 
  rate(decay_rate), 
  minimum(minimum_value)
{}

template <typename value_type>
value_type exponential_decay<value_type>::operator()(std::size_t k) const
{
    // v(k) = initial * rate^(k-1), but never below the minimum
    const value_type n = static_cast<value_type>(std::max<std::size_t>(k, 1) - 1);
    return std::max(minimum, initial * std::pow(rate, n));
}

template <typename value_type>
linear_decay<value_type>::linear_decay(value_type initial_value, 
                                       value_type last_value, 
                                       std::size_t step_count)
: initial(initial_value), 
  last(last_value), 
  steps(step_count)
{}

template <typename value_type>
value_type linear_decay<value_type>::operator()(std::size_t k) const
{
    // the fraction of the way from `initial` to `last`: 0 at k = 1, 1 from k = steps + 1
    const std::size_t done = std::min(std::max<std::size_t>(k, 1) - 1, steps);
    const value_type fraction = steps == 0 ? value_type(1) 
                                           : static_cast<value_type>(done) / static_cast<value_type>(steps);
    return initial + (last - initial) * fraction;
}

template <class state_class, 
          class action_class>
std::size_t visit_counter<state_class,action_class>::count(const state_class & s_t, 
                                                           const action_class & a_t) const
{
    auto state = counts_.find(s_t);
    if (state == counts_.end()) {
        return 0;
    }
    auto entry = state->second.find(a_t);
    return entry == state->second.end() ? 0 : entry->second;
}

template <class state_class, 
          class action_class>
std::size_t visit_counter<state_class,action_class>::increment(const state_class & s_t, 
                                                               const action_class & a_t)
{
    return ++counts_[s_t][a_t];
}

template <class state_class, 
          class action_class>
void visit_counter<state_class,action_class>::clear()
{
    counts_.clear();
}

template <class state_class, 
          class action_class,
          typename value_type>
value_type step_size(const std::function<value_type(std::size_t)> & learning_rate,
                     value_type alpha,
                     visit_counter<state_class,action_class> & visits,
                     const state_class & s_t,
                     const action_class & a_t)
{
    // no schedule: the same learning rate for every update
    if (!learning_rate) {
        return alpha;
    }
    // the n-th update of (s_t, a_t) uses the n-th learning rate of the schedule
    return learning_rate(visits.increment(s_t, a_t));
}
} // end of namespace
#endif
