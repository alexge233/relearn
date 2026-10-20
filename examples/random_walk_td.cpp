/**
 * The random walk learnt with TD(0) and n-step TD (Sutton & Barto, Example 6.2, Figures 6.2 and 7.2).
 *
 * 1. Example 6.2: the values TD(0) learns in the 5-state walk, after 0, 1, 10 and 100 walks,
 *    next to the true ones, 1/6 ... 5/6.
 * 2. Example 6.2's right-hand plot: the error of TD(0) and of constant-α Monte Carlo as they learn, averaged over
 *    100 runs. TD(0) learns faster, for every learning rate tried.
 * 3. Figure 7.2: the 19-state walk, learnt with n-step TD for n = 1, 2, 4, ... 512, and the error
 *    over the first 10 walks for each learning rate α. An intermediate n learns best.
 *
 * The program succeeds if TD(0) beats Monte Carlo, and if an intermediate n (2, 4 or 8) is best.
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
 * @brief the root mean square error of the learnt values
 * @param walk gives the true values
 * @param memory holds the learnt values: V(s) is the value of (s, 0)
 * @return sqrt(Σ_s (V(s) - v(s))² / number of states)
 */
double rms_error(const game_type & walk,
                 const types::policy & memory)
{
    double sum = 0;
    for (unsigned int s = 1; s <= walk.states(); s++) {
        double error = memory.value(types::state(s), types::action(0)) - walk.true_value(s);
        sum += error * error;
    }
    return std::sqrt(sum / walk.states());
}

/**
 * @brief start every value of the 5-state walk at 0.5, as in Example 6.2
 * @param memory receives the values
 */
void start_at_half(types::policy & memory)
{
    for (unsigned int s = 1; s <= 5; s++) {
        memory.update(types::state(s), types::action(0), 0.5);
    }
}

/**
 * @brief Example 6.2 (left): the values TD(0) learns, with α = 0.1, after 0, 1, 10 and 100 walks
 */
void example_6_2()
{
    game_type walk(5, 0, 1, 1);
    types::policy memory;
    start_at_half(memory);
    relearn::n_step_td<types::state, types::action> td(0.1, 1.0, 1);
    // the values after each of these numbers of walks
    const std::vector<unsigned int> shown = {0, 1, 10, 100};
    std::vector<std::vector<double>> values;
    unsigned int walked = 0;
    for (auto upto : shown) {
        for (; walked < upto; walked++) {
            auto r = games::play(walk, only_action, 100000);
            td(r.episode, memory, r.terminated);
        }
        std::vector<double> now;
        for (unsigned int s = 1; s <= 5; s++) {
            now.push_back(memory.value(types::state(s), types::action(0)));
        }
        values.push_back(now);
    }
    std::cout << "Example 6.2: the 5-state walk, TD(0) with α = 0.1" << std::endl;
    std::cout << "    walks:";
    for (auto upto : shown) {
        std::cout << std::setw(7) << upto;
    }
    std::cout << "    true" << std::endl;
    const char names[] = "ABCDE";
    for (unsigned int s = 1; s <= 5; s++) {
        std::cout << "        " << names[s - 1] << ":";
        for (const auto & now : values) {
            std::cout << std::setw(7) << now[s - 1];
        }
        std::cout << std::setw(8) << walk.true_value(s) << std::endl;
    }
}

/**
 * @brief Example 6.2, the right-hand plot: the RMS error after each walk, averaged over 100 runs
 * @param monte_carlo is true for constant-α Monte Carlo, false for TD(0)
 * @param alpha is the learning rate
 * @param walks is the number of walks of each run
 * @return the mean error after 0, 1, ... @param walks walks
 */
std::vector<double> figure_6_2(bool monte_carlo,
                               double alpha,
                               unsigned int walks)
{
    std::vector<double> error(walks + 1, 0);
    for (unsigned int run = 0; run < 100; run++) {
        game_type walk(5, 0, 1, run);
        types::policy memory;
        start_at_half(memory);
        // TD(0), and every-visit Monte Carlo with a constant learning rate (Section 6.1, equation 6.1)
        relearn::n_step_td<types::state, types::action> td(alpha, 1.0, 1);
        relearn::monte_carlo<types::state, types::action> mc(1.0, false);
        mc.learning_rate = [alpha](std::size_t) { return alpha; };
        error[0] += rms_error(walk, memory) / 100;
        for (unsigned int i = 1; i <= walks; i++) {
            auto r = games::play(walk, only_action, 100000);
            if (monte_carlo) {
                mc(r.episode, memory, r.terminated);
            }
            else {
                td(r.episode, memory, r.terminated);
            }
            error[i] += rms_error(walk, memory) / 100;
        }
    }
    return error;
}

/**
 * @brief Figure 7.2: the RMS error of the 19-state walk, over the first 10 walks and 100 runs
 * @param n is the number of steps
 * @param alpha is the learning rate
 * @return the error after each of the first 10 walks, averaged over them and over the runs
 */
double figure_7_2(unsigned int n,
                  double alpha)
{
    double error = 0;
    for (unsigned int run = 0; run < 100; run++) {
        // leaving on the left is worth -1, on the right +1, and every value starts at 0
        game_type walk(19, -1, 1, run);
        types::policy memory;
        relearn::n_step_td<types::state, types::action> td(alpha, 1.0, n);
        for (unsigned int i = 0; i < 10; i++) {
            auto r = games::play(walk, only_action, 100000);
            td(r.episode, memory, r.terminated);
            error += rms_error(walk, memory) / 1000;
        }
    }
    return error;
}

int main()
{
    std::cout << std::fixed << std::setprecision(3);

    // 1. Example 6.2: the values learnt by TD(0)
    example_6_2();

    // 2. Example 6.2's right-hand plot: TD(0) against constant-α Monte Carlo
    std::cout << std::endl << "Example 6.2 (right): the RMS error over the 5 states, averaged over 100 runs" << std::endl;
    const std::vector<unsigned int> shown = {0, 1, 10, 25, 50, 75, 100};
    std::cout << "    walks:            ";
    for (auto w : shown) {
        std::cout << std::setw(7) << w;
    }
    std::cout << std::endl;
    double best_td = std::numeric_limits<double>::infinity();
    double best_mc = std::numeric_limits<double>::infinity();
    for (int method = 0; method < 2; method++) {
        const bool mc = method == 1;
        const std::vector<double> rates = mc ? std::vector<double>{0.01, 0.02, 0.03, 0.04}
                                             : std::vector<double>{0.05, 0.1, 0.15};
        for (auto alpha : rates) {
            auto error = figure_6_2(mc, alpha, 100);
            std::cout << "    " << (mc ? "Monte Carlo" : "TD(0)      ")
                      << " α = " << std::setprecision(2) << alpha << std::setprecision(3) << ":";
            for (auto w : shown) {
                std::cout << std::setw(7) << error[w];
            }
            std::cout << std::endl;
            double & best = mc ? best_mc : best_td;
            best = std::min(best, error[100]);
        }
    }

    // 3. Figure 7.2: n-step TD on the 19-state walk, for each n and α
    std::cout << std::endl << "Figure 7.2: the 19-state walk, n-step TD, the RMS error over the first 10 walks"
              << " (100 runs)" << std::endl;
    const std::vector<unsigned int> steps = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    std::cout << "        α   n =";
    for (auto n : steps) {
        std::cout << std::setw(6) << n;
    }
    std::cout << std::endl;
    std::vector<double> lowest(steps.size(), std::numeric_limits<double>::infinity());
    for (int i = 0; i <= 20; i++) {
        const double alpha = 0.05 * i;
        std::cout << "    " << std::setprecision(2) << std::setw(5) << alpha << "     " << std::setprecision(3);
        for (std::size_t j = 0; j < steps.size(); j++) {
            double error = figure_7_2(steps[j], alpha);
            lowest[j] = std::min(lowest[j], error);
            std::cout << std::setw(6) << std::setprecision(2) << error;
        }
        std::cout << std::endl;
    }
    std::cout << "    lowest    ";
    std::size_t best = 0;
    for (std::size_t j = 0; j < steps.size(); j++) {
        std::cout << std::setw(6) << std::setprecision(3) << lowest[j];
        if (lowest[j] < lowest[best]) {
            best = j;
        }
    }
    std::cout << std::endl << "    the best n is " << steps[best] << std::endl;

    // TD(0) must learn faster than Monte Carlo, and an intermediate n must be best
    bool learnt = (best_td < best_mc) && (steps[best] >= 2 && steps[best] <= 8);
    return learnt ? EXIT_SUCCESS : EXIT_FAILURE;
}
