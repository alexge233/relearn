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
/** 
 
   ________  _______   ___       _______   ________  ________  ________      
  |\   __  \|\  ___ \ |\  \     |\  ___ \ |\   __  \|\   __  \|\   ___  \
  \ \  \|\  \ \   __/|\ \  \    \ \   __/|\ \  \|\  \ \  \|\  \ \  \\ \  \
   \ \   _  _\ \  \_|/_\ \  \    \ \  \_|/_\ \   __  \ \   _  _\ \  \\ \  \
    \ \  \\  \\ \  \_|\ \ \  \____\ \  \_|\ \ \  \ \  \ \  \\  \\ \  \\ \  \
     \ \__\\ _\\ \_______\ \_______\ \_______\ \__\ \__\ \__\\ _\\ \__\\ \__\
      \|__|\|__|\|_______|\|_______|\|_______|\|__|\|__|\|__|\|__|\|__| \|__|
 

 * @brief relearn C++ reinforcement learning library
 * @version 0.1.1
 * @date 07-July-2017
 * @author Alex Giokas <alexge233@hotmail.com>
 *
 * @note: do not confuse `state_class` with `state_trait`
 * and similarly, `action_class` with `action_trait`.
 * Whereas `state_class` refers to the template parameter being used
 * in class `policy` the `state_trait` is what characterises a `state_cass`,
 * or similarly an action. Essentially, the template parameter `state_class` 
 * used later in policy, refers to template class `state` wrapper but because this
 * is a template-based library, you must roll out your own `state` class(es) provided
 * that it:
 *	- is hashable (e.g., can be used with `std::hash` or exposes a `hash()` method
 *  - is comparable (e.g., can be used with `operator<`)
 * Similary `action` takes the same approach.
 *
 * The library is header-only, in several headers: this one includes them all.
 * To use only part of it, include the parts you need from `relearn/`, e.g. `relearn/q_learning.hpp`.
 */
#include "relearn/core.hpp"
#include "relearn/schedules.hpp"
#include "relearn/q_learning.hpp"
#include "relearn/sarsa.hpp"
#include "relearn/double_q_learning.hpp"
#include "relearn/dyna.hpp"
#include "relearn/monte_carlo.hpp"
#include "relearn/mcts.hpp"
#include "relearn/n_step.hpp"
#include "relearn/policy_gradient.hpp"
#include "relearn/r_learning.hpp"
#include "relearn/prioritized_sweeping.hpp"
#include "relearn/dynamic_programming.hpp"
#include "relearn/bandits.hpp"
#endif
