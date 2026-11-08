/**
 * The racetrack (Sutton & Barto, Exercise 5.12 and Example 8.6), planned with its model (Sections 8.7, 8.9 and 8.10).
 *
 * 1. Example 8.6: value iteration against real-time dynamic programming (RTDP). Value iteration sweeps every state
 *    until no value changes by 1e-4; RTDP only updates the states its greedy trials from the start line meet, and
 *    stops (checked every 100 trials) once its greedy policy, evaluated exactly, is within 0.05 steps of the optimal
 *    one. RTDP needs fewer updates, and leaves most states alone.
 * 2. Planning from rough values, those of a few sweeps of value iteration from 0: the greedy policy on them, heuristic
 *    search 2 and 3 moves deep (Section 8.9), and rollouts (Section 8.10) of that greedy policy, 20 for each action,
 *    each cut after 200 steps. Each one is a policy, evaluated exactly from the start line ("never" if it can never
 *    finish). Looking further ahead, or rolling the rough policy out, plays better than the rough values alone.
 *
 * The program succeeds if RTDP reaches the optimal policy with fewer updates than value iteration, updating fewer than
 * half the states, if from 8 sweeps heuristic search 3 moves deep beats the greedy policy, and if from 3 sweeps the
 * rollouts do.
 *
 * Run it with `--watch` to see the optimal policy drive five races, noise and all, once value iteration has found it
 * (have a look at docs/watch.md).
 *
 * @see games/racetrack.hpp for the rules
 * @see docs/planning.md for the algorithms
 */
#include "games/racetrack.hpp"
#include "games/views.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <vector>

// the frames go to the terminal, or, built by the window sub-project, to a raylib window
#ifdef RELEARN_WINDOW
#include "window/window.hpp"
using screen = games::window_screen;
#else
using screen = games::terminal_screen;
#endif
// set shortcuts to the game and to its relearn types
using game_type = games::racetrack;
using types     = games::types<game_type>;
using planner   = relearn::dynamic_programming<types::state, types::action>;

/**
 * @brief evaluate a policy exactly, from the start line
 * @param track is the racetrack, @param model its model
 * @param choose gives the action of the policy in a state; it is asked only in the states the policy can reach
 * @return the mean value of the start line under the policy: minus the expected number of steps to the finish,
 *         or minus infinity if the policy can never reach the finish line
 */
double start_value(const game_type & track,
                   const game_type::model_type & model,
                   const std::function<types::action(const types::state &)> & choose)
{
    // the states the policy can reach from the start line, and its action in each of them
    planner::policy_type pi;
    std::map<unsigned int, std::vector<unsigned int>> before;
    std::vector<types::state> todo;
    for (auto s : track.starts()) {
        todo.push_back(types::state(s));
    }
    while (!todo.empty()) {
        auto s = todo.back();
        todo.pop_back();
        if (model.actions(s).empty() || pi.count(s)) {
            continue;
        }
        auto a = choose(s);
        pi.emplace(s, a);
        for (const auto & o : model.outcomes(s, a)) {
            before[o.next.trait()].push_back(s.trait());
            todo.push_back(o.next);
        }
    }
    // the states the finish line can be reached from, backwards from it: if any state the policy reaches is not one
    // of them, the policy can go round for ever from there, and its evaluation would only stop at its last sweep
    std::map<unsigned int, bool> finishes = {{track.finished(), true}};
    std::vector<unsigned int> back = {track.finished()};
    while (!back.empty()) {
        auto s = back.back();
        back.pop_back();
        for (auto p : before[s]) {
            if (!finishes[p]) {
                finishes[p] = true;
                back.push_back(p);
            }
        }
    }
    for (const auto & entry : pi) {
        if (!finishes[entry.first.trait()]) {
            return -std::numeric_limits<double>::infinity();
        }
    }
    planner exact(1.0, 1e-9);
    exact.max_sweeps = 5000;
    exact.evaluate(model, pi);
    double v = 0;
    for (auto s : track.starts()) {
        v += exact.value(types::state(s)) / static_cast<double>(track.starts().size());
    }
    return v;
}

/// @return the first action of @param s whose q-value under @param q is within 1e-9 of the highest
template <class values>
types::action best(const game_type::model_type & model,
                   const types::state & s,
                   const values & q)
{
    auto actions = model.actions(s);
    double top = q(s, actions[0]);
    for (const auto & a : actions) {
        top = std::max(top, q(s, a));
    }
    std::size_t b = 0;
    while (q(s, actions[b]) < top - 1e-9) {
        b++;
    }
    return actions[b];
}

int main(int argc,
         char ** argv)
{
    const auto options = games::watch_options::parse(argc, argv);
    std::cout << std::fixed;
    game_type track;
    auto model = track.model();
    std::size_t live = 0;
    for (const auto & s : model.states) {
        live += model.actions(s).empty() ? 0 : 1;
    }

    // 1. Example 8.6: value iteration against RTDP
    planner sweeps(1.0, 1e-4);
    sweeps.value_iteration(model);
    const double optimal = start_value(track, model, [&](const types::state & s) {
        return best(model, s, [&](const types::state & x, const types::action & a) { return sweeps.q_value(model, x, a); });
    });
    std::cout << "Example 8.6: the racetrack, " << live << " states, value iteration against real-time dynamic programming" << std::endl;
    std::cout << "    the optimal policy: " << std::setprecision(3) << -optimal << " steps from the start line, on average" << std::endl;
    std::cout << "    value iteration: " << sweeps.sweeps() << " sweeps, " << sweeps.sweeps() * live << " updates, of every state" << std::endl;
    double mean_trials = 0;
    double mean_updates = 0;
    double mean_states = 0;
    const unsigned int runs = 10;
    for (unsigned int run = 0; run < runs; run++) {
        relearn::real_time_dynamic_programming<types::state, types::action> rtdp(1.0);
        std::mt19937 prng(run);
        std::uniform_int_distribution<std::size_t> pick(0, track.starts().size() - 1);
        std::size_t trials = 0;
        while (trials < 100000) {
            rtdp.trial(model, types::state(track.starts()[pick(prng)]), prng);
            trials++;
            if (trials % 100 == 0) {
                const double greedy = start_value(track, model, [&](const types::state & s) {
                    return best(model, s, [&](const types::state & x, const types::action & a) { return rtdp.q_value(model, x, a); });
                });
                if (greedy > optimal - 0.05) {
                    break;
                }
            }
        }
        mean_trials += static_cast<double>(trials) / runs;
        mean_updates += static_cast<double>(rtdp.updates()) / runs;
        mean_states += static_cast<double>(rtdp.values.size()) / runs;
    }
    std::cout << "    RTDP: " << std::setprecision(0) << mean_trials << " trials, " << mean_updates << " updates, of "
              << mean_states << " states (" << std::setprecision(1) << 100 * mean_states / live
              << "%), until its greedy policy is within 0.05 steps of the optimal one (" << runs << " runs)" << std::endl;

    // watched: the optimal policy drives five races, with the noise of the game
    if (options.on) {
        games::watched<game_type, screen> car{game_type(track.default_track(), true, 3)};
        options.apply(car);
        car.title = "racetrack, the optimal policy";
        auto driver = [&](const unsigned int & s, const std::vector<unsigned int> &) {
            return best(model, types::state(s), [&](const types::state & x, const types::action & a) {
                return sweeps.q_value(model, x, a);
            }).trait();
        };
        for (int race = 0; race < 5; race++) {
            games::play(car, driver, 1000);
        }
    }

    // 2. planning from the values of a few sweeps
    std::cout << std::endl << "Planning from rough values (a few sweeps of value iteration from 0): steps from the start line" << std::endl;
    std::cout << "    sweeps     greedy   search 2   search 3   rollouts" << std::endl;
    double greedy_3 = 0;
    double rolled_3 = 0;
    double greedy_8 = 0;
    double deep_8 = 0;
    for (unsigned int k : {3u, 8u}) {
        planner rough(1.0, 0);
        rough.in_place = false;
        rough.max_sweeps = k;
        rough.value_iteration(model);
        auto rough_q = [&](const types::state & x, const types::action & a) { return rough.q_value(model, x, a); };
        std::vector<double> steps;
        for (unsigned int depth : {1u, 2u, 3u}) {
            relearn::heuristic_search<types::state, types::action> search(1.0, depth);
            search.values = rough.values;
            steps.push_back(-start_value(track, model, [&](const types::state & s) { return search.choose(model, s); }));
        }
        // the rollout policy: greedy on the rough values
        relearn::rollout_planner<types::state, types::action> rollouts(1.0, 20,
            [&](const types::state & s, const std::vector<types::action> &, std::mt19937 &) { return best(model, s, rough_q); });
        rollouts.max_steps = 200;
        std::mt19937 prng(7);
        steps.push_back(-start_value(track, model, [&](const types::state & s) { return rollouts.choose(model, s, prng); }));
        std::cout << "    " << std::setw(6) << k << std::setprecision(2);
        for (auto v : steps) {
            if (std::isinf(v)) {
                std::cout << std::setw(11) << "never";
            }
            else {
                std::cout << std::setw(11) << v;
            }
        }
        std::cout << std::endl;
        if (k == 3) {
            greedy_3 = steps[0];
            rolled_3 = steps[3];
        }
        else {
            greedy_8 = steps[0];
            deep_8 = steps[2];
        }
    }
    std::cout << "    optimal" << std::setw(10) << -optimal << std::endl;

    const bool real_time = mean_trials < 100000 && mean_updates < static_cast<double>(sweeps.sweeps() * live) && mean_states < live / 2.0;
    const bool planning = deep_8 < greedy_8 && rolled_3 < greedy_3;
    return (real_time && planning) ? EXIT_SUCCESS : EXIT_FAILURE;
}
