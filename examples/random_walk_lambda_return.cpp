/**
 * The 19-state random walk learnt with the λ-return algorithms (Sutton & Barto, Sections 12.3, 12.4 and 12.8).
 *
 * 1. Truncated TD(λ): the error over the first 10 walks, at the best α, for horizons n = 1 to 1000, next to
 *    the online λ-return algorithm, averaged over 100 runs. n = 1 is TD(0), and n = 1000 (longer than any walk)
 *    is the offline λ-return algorithm, which only learns once the walk has ended. A short horizon learns
 *    sooner, and with a large λ it is better than both.
 * 2. The online λ-return algorithm and true online TD(λ), learning after every step of the same walks:
 *    one redoes the whole walk so far at every step, the other costs one step of TD(λ), and they learn
 *    the very same values.
 * 3. A γ that varies (Section 12.8): with γ = 0 in state 5, the return stops whenever the walk gets there,
 *    so each state learns what the walk earns before it next reaches state 5. Right of it, that is the chance
 *    of leaving on the right first, (s - 5) / 15; left of it, minus the chance of leaving on the left first,
 *    -(5 - s) / 5; and in state 5 itself, -1/15.
 *
 * The program succeeds if, with λ = 0.9, a horizon of 2 to 16 is better than both n = 1 and n = 1000, if the
 * online λ-return algorithm is better than the offline one for every λ, if it learns the values of true online
 * TD(λ) to within 1e-12, and if the values learnt with γ = 0 in state 5 are within 0.05 of the true ones.
 *
 * @see games/random_walk.hpp for the rules
 * @see docs/eligibility_traces.md for the algorithms
 */
#include "games/random_walk.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

// set shortcuts to the game and to its relearn types
using game_type = games::random_walk;
using types     = games::types<game_type>;

/**
 * @brief the agent of a prediction task: there is only one action, so it always takes it
 * @return action 0
 */
unsigned int only_action(const unsigned int &,
                         const std::vector<unsigned int> &)
{
    return 0u;
}

/**
 * @brief the root mean square error of the learnt values, at most 1
 * @param walk gives the number of states
 * @param memory holds the learnt values: V(s) is the value of (s, 0)
 * @param truth gives the true value of each state
 * @return sqrt(Σ_s (V(s) - v(s))² / number of states), or 1 if the values have blown up
 */
template <class true_values>
double rms_error(const game_type & walk,
                 const types::policy & memory,
                 true_values truth)
{
    double sum = 0;
    for (unsigned int s = 1; s <= walk.states(); s++) {
        double error = memory.value(types::state(s), types::action(0)) - truth(s);
        sum += error * error;
    }
    double rms = std::sqrt(sum / walk.states());
    return (std::isfinite(rms) && rms < 1) ? rms : 1;
}

/**
 * @brief the lowest error of a learner over the first 10 walks, over α = 0.05, 0.1, ... 1 (100 runs)
 * @param make makes the learner, given α
 * @return the error after each of the first 10 walks, averaged over them and over the runs, at the best α
 */
template <class maker>
double lowest_error(maker make)
{
    double lowest = std::numeric_limits<double>::infinity();
    for (int i = 1; i <= 20; i++) {
        double error = 0;
        for (unsigned int run = 0; run < 100; run++) {
            // leaving on the left is worth -1, on the right +1, and every value starts at 0
            game_type walk(19, -1, 1, run);
            types::policy memory;
            auto learner = make(0.05 * i);
            for (unsigned int k = 0; k < 10; k++) {
                auto r = games::play(walk, only_action, 100000);
                learner(r.episode, memory, r.terminated);
                error += rms_error(walk, memory, [&](unsigned int s) { return walk.true_value(s); }) / 1000;
            }
        }
        lowest = std::min(lowest, error);
    }
    return lowest;
}

int main()
{
    std::cout << std::fixed << std::setprecision(4);

    // 1. truncated TD(λ), by horizon, and the online λ-return algorithm
    std::cout << "Truncated TD(λ) on the 19-state walk: the RMS error over the first 10 walks, at the best α (100 runs)" << std::endl;
    const std::vector<unsigned int> horizons = {1, 2, 4, 8, 16, 1000};
    const std::vector<double> lambdas = {0.4, 0.8, 0.9};
    std::cout << "        λ   n =";
    for (auto n : horizons) {
        std::cout << std::setw(8) << n;
    }
    std::cout << "   online λ-return" << std::endl;
    bool short_is_best = false;
    bool online_is_better = true;
    for (auto lambda : lambdas) {
        std::cout << "    " << std::setprecision(1) << std::setw(5) << lambda << "     " << std::setprecision(4);
        std::vector<double> errors;
        for (auto n : horizons) {
            errors.push_back(lowest_error([lambda, n](double alpha) {
                return relearn::truncated_td_lambda<types::state, types::action>(alpha, 1.0, lambda, n);
            }));
            std::cout << std::setw(8) << errors.back();
        }
        const double online = lowest_error([lambda](double alpha) {
            return relearn::online_lambda_return<types::state, types::action>(alpha, 1.0, lambda);
        });
        std::cout << std::setw(18) << online << std::endl;
        // with λ = 0.9, the best of n = 2 ... 16 is better than TD(0) and than the offline λ-return
        if (lambda == 0.9) {
            const double best_short = *std::min_element(errors.begin() + 1, errors.end() - 1);
            short_is_best = best_short < errors.front() && best_short < errors.back();
        }
        online_is_better = online_is_better && online < errors.back();
    }

    // 2. the online λ-return algorithm and true online TD(λ), learning after every step of the same walks
    double largest = 0;
    for (unsigned int run = 0; run < 100; run++) {
        game_type walk(19, -1, 1, run);
        game_type same(19, -1, 1, run);
        types::policy redone, true_online;
        relearn::online_lambda_return<types::state, types::action> forward(0.4, 1.0, 0.8);
        relearn::true_online_td_lambda<types::state, types::action> backward(0.4, 1.0, 0.8);
        for (unsigned int k = 0; k < 10; k++) {
            games::play_online(walk, only_action, forward, redone, 100000);
            games::play_online(same, only_action, backward, true_online, 100000);
            for (unsigned int s = 1; s <= walk.states(); s++) {
                largest = std::max(largest, std::fabs(redone.value(types::state(s), types::action(0)) -
                                                      true_online.value(types::state(s), types::action(0))));
            }
        }
    }
    std::cout << std::endl << "The online λ-return algorithm and true online TD(λ), learning after every step of the same walks"
              << std::endl << "(α = 0.4, λ = 0.8, 100 runs of 10 walks): the largest difference of their values is "
              << std::scientific << std::setprecision(1) << largest << std::endl << std::fixed;

    // 3. γ = 0 in state 5: the return stops there
    auto truth = [](unsigned int s) { return s > 5 ? (s - 5.0) / 15 : (s < 5 ? -(5.0 - s) / 5 : -1.0 / 15); };
    auto gamma_of = [](const types::state & s) { return s.trait() == 5 ? 0.0 : 1.0; };
    std::cout << std::endl << "γ = 0 in state 5: true online TD(λ) after 1000 walks (α = 0.02, λ = 0.8)" << std::endl;
    std::cout << "    state   learnt    true" << std::endl;
    double mean = 0;
    for (unsigned int run = 0; run < 10; run++) {
        game_type walk(19, -1, 1, run);
        types::policy memory;
        relearn::true_online_td_lambda<types::state, types::action> td(0.02, 1.0, 0.8);
        td.gamma_of = gamma_of;
        for (unsigned int k = 0; k < 1000; k++) {
            auto r = games::play(walk, only_action, 100000);
            td(r.episode, memory, r.terminated);
        }
        mean += rms_error(walk, memory, truth) / 10;
        // the values of the first run
        if (run == 0) {
            for (unsigned int s = 1; s <= walk.states(); s++) {
                std::cout << "    " << std::setw(5) << s << std::setprecision(4)
                          << std::setw(9) << memory.value(types::state(s), types::action(0))
                          << std::setw(8) << truth(s) << std::endl;
            }
        }
    }
    std::cout << "    the RMS error, over 10 runs: " << mean << std::endl;

    bool learnt = short_is_best && online_is_better && (largest < 1e-12) && (mean < 0.05);
    return learnt ? EXIT_SUCCESS : EXIT_FAILURE;
}
