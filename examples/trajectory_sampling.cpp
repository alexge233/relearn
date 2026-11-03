/**
 * Trajectory sampling (Sutton & Barto, Section 8.6, Figure 8.8): expected updates spread uniformly over every
 * state/action pair, against the same updates along trajectories that follow the current policy (on-policy).
 *
 * The tasks are those of the book: 1000 states, 2 actions, b branches for each pair, a 0.1 chance of ending at every
 * step, and rewards drawn from a normal distribution (see games/branching_task.hpp). After 250 to 20000 expected
 * updates, the greedy policy on the values is evaluated exactly, from the start state, and averaged over 30 tasks.
 * On-policy updates find a good policy for the start state sooner, all the more so with few branches; uniform updates
 * catch up later, and pass them.
 *
 * The program succeeds if, after 250 updates, the on-policy updates are better than the uniform ones for b = 1 and 3,
 * and if after 20000 updates the uniform ones are better for b = 3 and 10.
 *
 * @see games/branching_task.hpp for the tasks
 * @see docs/planning.md for the algorithms
 */
#include "games/branching_task.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using task_type = games::branching_task;
using types     = games::types<task_type>;
using planner   = relearn::dynamic_programming<types::state, types::action>;

/// @return the value of the start state of @param task under the greedy policy on the values of @param q, worked out exactly
double start_value(const task_type & task,
                   const task_type::model_type & model,
                   const types::policy & q)
{
    planner::policy_type greedy;
    for (const auto & s : model.states) {
        if (!model.actions(s).empty()) {
            const bool first = q.value(s, types::action(0u)) >= q.value(s, types::action(1u));
            greedy.emplace(s, types::action(first ? 0u : 1u));
        }
    }
    planner exact(1.0, 1e-9);
    exact.evaluate(model, greedy);
    return exact.value(types::state(task.start()));
}

int main()
{
    std::cout << std::fixed << std::setprecision(3);
    const std::vector<std::size_t> marks = {250, 1000, 2000, 5000, 10000, 20000};
    const unsigned int tasks = 30;
    std::cout << "Figure 8.8: 1000 states, the value of the start state under the greedy policy, after each number of"
              << " expected updates (" << tasks << " tasks)" << std::endl;
    std::cout << "    updates:          ";
    for (auto m : marks) {
        std::cout << std::setw(8) << m;
    }
    std::cout << std::endl;
    std::vector<std::vector<double>> uniform_values, on_policy_values;
    for (unsigned int b : {1u, 3u, 10u}) {
        std::vector<double> uniform(marks.size(), 0), on_policy(marks.size(), 0);
        for (unsigned int k = 0; k < tasks; k++) {
            task_type task(1000, b, k);
            auto model = task.model();
            for (auto how : {relearn::update_distribution::uniform, relearn::update_distribution::on_policy}) {
                relearn::trajectory_sampling<types::state, types::action> sampling(1.0, how, 0.1);
                types::policy q;
                std::mt19937 prng(100 + k);
                std::size_t done = 0;
                for (std::size_t m = 0; m < marks.size(); m++) {
                    sampling.plan(model, q, types::state(task.start()), marks[m] - done, prng);
                    done = marks[m];
                    auto & row = how == relearn::update_distribution::uniform ? uniform : on_policy;
                    row[m] += start_value(task, model, q) / tasks;
                }
            }
        }
        std::cout << "    b = " << std::setw(2) << b << ", uniform    ";
        for (auto v : uniform) {
            std::cout << std::setw(8) << v;
        }
        std::cout << std::endl << "    b = " << std::setw(2) << b << ", on-policy  ";
        for (auto v : on_policy) {
            std::cout << std::setw(8) << v;
        }
        std::cout << std::endl;
        uniform_values.push_back(uniform);
        on_policy_values.push_back(on_policy);
    }

    // on-policy first for b = 1 and 3, uniform last for b = 3 and 10
    const bool early = on_policy_values[0].front() > uniform_values[0].front() &&
                       on_policy_values[1].front() > uniform_values[1].front();
    const bool late = uniform_values[1].back() > on_policy_values[1].back() &&
                      uniform_values[2].back() > on_policy_values[2].back();
    return (early && late) ? EXIT_SUCCESS : EXIT_FAILURE;
}
