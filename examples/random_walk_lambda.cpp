/**
 * The 19-state random walk learnt with TD(λ) (Sutton & Barto, Example 7.1, Sections 12.1 and 12.2).
 *
 * 1. Figure 12.6: the error of TD(λ) over the first 10 walks, for each λ and learning rate α,
 *    averaged over 100 runs. An intermediate λ learns best, as an intermediate n does for n-step TD.
 * 2. Offline TD(λ), which changes the values only at the end of each walk, next to the λ-return
 *    algorithm worked out from its definition: with accumulating traces, they learn the same values.
 *
 * The program succeeds if an intermediate λ (0.4 to 0.95) is best, and if offline TD(λ)
 * and the λ-return algorithm agree.
 *
 * @see games/random_walk.hpp for the rules
 */
#include "games/random_walk.hpp"
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
 * @param walk gives the true values
 * @param memory holds the learnt values: V(s) is the value of (s, 0)
 * @return sqrt(Σ_s (V(s) - v(s))² / number of states), or 1 if the values have blown up
 */
double rms_error(const game_type & walk,
                 const types::policy & memory)
{
    double sum = 0;
    for (unsigned int s = 1; s <= walk.states(); s++) {
        double error = memory.value(types::state(s), types::action(0)) - walk.true_value(s);
        sum += error * error;
    }
    double rms = std::sqrt(sum / walk.states());
    return (std::isfinite(rms) && rms < 1) ? rms : 1;
}

/**
 * @brief Figure 12.6: the RMS error of TD(λ) on the 19-state walk, over the first 10 walks and 100 runs
 * @param lambda is λ
 * @param alpha is the learning rate
 * @return the error after each of the first 10 walks, averaged over them and over the runs
 */
double figure_12_6(double lambda,
                   double alpha)
{
    double error = 0;
    for (unsigned int run = 0; run < 100; run++) {
        // leaving on the left is worth -1, on the right +1, and every value starts at 0
        game_type walk(19, -1, 1, run);
        types::policy memory;
        relearn::td_lambda<types::state, types::action> td(alpha, 1.0, lambda);
        for (unsigned int i = 0; i < 10; i++) {
            auto r = games::play(walk, only_action, 100000);
            td(r.episode, memory, r.terminated);
            error += rms_error(walk, memory) / 1000;
        }
    }
    return error;
}

/**
 * @brief the λ-return algorithm, offline (Section 12.1), worked out from its definition for one walk
 * @param walk_episode is a finished walk
 * @param memory holds the values: each V(s_t) moves by α (G^λ_t - V(s_t)), all of them worked out
 *        with the values from before the walk, and added up at its end
 * @param alpha is the learning rate, @param lambda is λ (and γ = 1)
 *
 * G^λ_t = (1 - λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_{t:t+n} + λ^{T-t-1} G_t, where G_{t:t+n} is the n-step
 * return of Section 7.1 and G_t the return to the end.
 */
void lambda_return(const types::episode & walk_episode,
                   types::policy & memory,
                   double alpha,
                   double lambda)
{
    const auto before = memory;
    const std::size_t T = walk_episode.size();
    for (std::size_t t = 0; t < T; t++) {
        // add up the n-step returns, weighted by (1 - λ) λ^{n-1}, and the rest of the weight on G_t
        double G = 0;
        double rewards = 0;
        double weight = 1;
        for (std::size_t k = t; k < T; k++) {
            rewards += walk_episode[k].state.reward();
            if (k + 1 < T) {
                const auto & later = walk_episode[k + 1];
                G += (1 - lambda) * weight * (rewards + before.value(later.state, later.action));
                weight *= lambda;
            }
            else {
                G += weight * rewards;
            }
        }
        // V(s_t) moves by α (G^λ_t - V(s_t))
        const auto & s = walk_episode[t];
        double v = before.value(s.state, s.action);
        memory.update(s.state, s.action, memory.value(s.state, s.action) + alpha * (G - v));
    }
}

int main()
{
    std::cout << std::fixed;

    // 1. Figure 12.6: TD(λ) on the 19-state walk, for each λ and α
    std::cout << "Figure 12.6: the 19-state walk, TD(λ), the RMS error over the first 10 walks (100 runs)" << std::endl;
    const std::vector<double> lambdas = {0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1};
    std::cout << "        α   λ =";
    for (auto lambda : lambdas) {
        std::cout << std::setprecision(3) << std::setw(7) << lambda;
    }
    std::cout << std::endl;
    std::vector<double> lowest(lambdas.size(), std::numeric_limits<double>::infinity());
    for (int i = 1; i <= 20; i++) {
        const double alpha = 0.05 * i;
        std::cout << "    " << std::setprecision(2) << std::setw(5) << alpha << "     ";
        for (std::size_t j = 0; j < lambdas.size(); j++) {
            double error = figure_12_6(lambdas[j], alpha);
            lowest[j] = std::min(lowest[j], error);
            std::cout << std::setw(7) << error;
        }
        std::cout << std::endl;
    }
    std::cout << "    lowest    ";
    std::size_t best = 0;
    for (std::size_t j = 0; j < lambdas.size(); j++) {
        std::cout << std::setprecision(3) << std::setw(7) << lowest[j];
        if (lowest[j] < lowest[best]) {
            best = j;
        }
    }
    std::cout << std::endl << "    the best λ is " << lambdas[best] << std::endl;

    // 2. offline TD(λ) and the λ-return algorithm, on the same 10 walks, with α = 0.1 and λ = 0.8
    game_type walk(19, -1, 1, 1);
    types::policy traced;
    types::policy forward;
    relearn::td_lambda<types::state, types::action> td(0.1, 1.0, 0.8);
    td.offline = true;
    for (unsigned int i = 0; i < 10; i++) {
        auto r = games::play(walk, only_action, 100000);
        td(r.episode, traced, r.terminated);
        lambda_return(r.episode, forward, 0.1, 0.8);
    }
    std::cout << std::endl << "Offline TD(λ) and the λ-return algorithm, after 10 walks (α = 0.1, λ = 0.8)" << std::endl;
    std::cout << "    state   TD(λ)   λ-return    true" << std::endl;
    double largest = 0;
    for (unsigned int s = 1; s <= walk.states(); s++) {
        double a = traced.value(types::state(s), types::action(0));
        double b = forward.value(types::state(s), types::action(0));
        largest = std::max(largest, std::fabs(a - b));
        std::cout << "    " << std::setw(5) << s << std::setprecision(4)
                  << std::setw(8) << a << std::setw(11) << b
                  << std::setw(8) << walk.true_value(s) << std::endl;
    }
    std::cout << std::scientific << std::setprecision(1)
              << "    the largest difference is " << largest << std::endl;

    // an intermediate λ must be best, and offline TD(λ) must be the λ-return algorithm
    bool learnt = (lambdas[best] >= 0.4 && lambdas[best] <= 0.95) && (largest < 1e-9);
    return learnt ? EXIT_SUCCESS : EXIT_FAILURE;
}
