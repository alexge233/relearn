/**
 * Dynamic programming (Sutton & Barto, Chapter 4), with `dynamic_programming`.
 *
 * Dynamic programming does not play: it is given the rules as a model, p(s', r | s, a),
 * and works out the values from it, sweep after sweep.
 *
 * 1. Figure 4.1: the values of the random policy in the 4 x 4 gridworld, after k sweeps,
 *    and the greedy policy on them, which is optimal long before the values have settled.
 * 2. Policy iteration and value iteration find the same optimal values.
 * 3. Example 4.3 and Figure 4.3: the gambler's problem, solved by value iteration,
 *    with the values after a few sweeps, and an optimal policy.
 *
 * The program succeeds if the values match the book's: -14, -20, -22, ... for the random policy,
 * V(25) = 0.16, V(50) = 0.4 and V(75) = 0.64 for the gambler, and if policy iteration and
 * value iteration agree.
 *
 * @see games/small_gridworld.hpp and games/gamblers_problem.hpp for the rules and their models
 */
#include "games/small_gridworld.hpp"
#include "games/gamblers_problem.hpp"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// set shortcuts to the states, the actions and the solver (both games use unsigned integers)
using state_type  = relearn::state<unsigned int>;
using action_type = relearn::action<unsigned int>;
using solver_type = relearn::dynamic_programming<state_type, action_type>;

/// @return π(a | s) = 1/4 for each of the four moves: the random policy
double random_policy(const state_type &,
                     const action_type &)
{
    return 0.25;
}

/// @return the arrow of the move @param a
char arrow(unsigned int a)
{
    switch (a) {
        case games::up:
            return '^';
        case games::right:
            return '>';
        case games::down:
            return 'v';
        default:
            return '<';
    }
}

/**
 * @brief print the values of the gridworld, next to the greedy policy on them
 * @param title says how the values were found
 * @param dp holds the values
 * @param model is the gridworld's model
 */
void print_grid(const std::string & title,
                const solver_type & dp,
                const games::small_gridworld::model_type & model)
{
    std::cout << title << std::endl;
    auto pi = dp.greedy(model);
    for (unsigned int y = 0; y < 4; y++) {
        std::cout << "   ";
        for (unsigned int x = 0; x < 4; x++) {
            std::cout << std::setw(7) << dp.value(state_type(y * 4 + x));
        }
        std::cout << "      ";
        for (unsigned int x = 0; x < 4; x++) {
            const unsigned int s = y * 4 + x;
            std::cout << ' ' << (games::small_gridworld::terminal(s) ? 'T' : arrow(pi.at(state_type(s)).trait()));
        }
        std::cout << std::endl;
    }
}

int main()
{
    std::cout << std::fixed << std::setprecision(2);
    bool learnt = true;

    // 1. Figure 4.1: the random policy, γ = 1, every value from the sweep before (two arrays)
    games::small_gridworld gridworld;
    auto grid = gridworld.model();
    std::cout << "the 4 x 4 gridworld: the values of the random policy after k sweeps," << std::endl
              << "and the greedy policy on them (Figure 4.1)" << std::endl;
    for (std::size_t k : {1u, 2u, 3u, 10u}) {
        solver_type dp(1.0, 1e-9);
        dp.in_place = false;
        dp.max_sweeps = k;
        dp.evaluate(grid, random_policy);
        print_grid("k = " + std::to_string(k) + ":", dp, grid);
    }
    solver_type random_values(1.0, 1e-9);
    random_values.in_place = false;
    random_values.evaluate(grid, random_policy);
    print_grid("k = infinity (" + std::to_string(random_values.sweeps()) + " sweeps until no value changes by 1e-9):",
               random_values, grid);
    // the values the book gives
    const std::vector<double> book = {  0, -14, -20, -22,
                                      -14, -18, -20, -20,
                                      -20, -20, -18, -14,
                                      -22, -20, -14,   0};
    for (unsigned int s = 0; s < 16; s++) {
        learnt = learnt && std::fabs(random_values.value(state_type(s)) - book[s]) < 1e-6;
    }
    // in place, each update uses the newest values, and the values settle in fewer sweeps
    solver_type in_place(1.0, 1e-9);
    in_place.evaluate(grid, random_policy);
    std::cout << "in place, the same values settle in " << in_place.sweeps() << " sweeps" << std::endl;

    // 2. policy iteration and value iteration, γ = 1
    std::cout << std::endl << "the optimal values, by policy iteration and by value iteration:" << std::endl;
    solver_type by_policy(1.0, 1e-9);
    by_policy.policy_iteration(grid, random_values.greedy(grid));
    print_grid("policy iteration, from the greedy policy above (evaluations: " + std::to_string(by_policy.iterations()) +
               ", sweeps: " + std::to_string(by_policy.sweeps()) + ")", by_policy, grid);
    solver_type by_values(1.0, 1e-9);
    by_values.value_iteration(grid);
    print_grid("value iteration (sweeps: " + std::to_string(by_values.sweeps()) + ")", by_values, grid);
    for (unsigned int s = 0; s < 16; s++) {
        learnt = learnt && std::fabs(by_policy.value(state_type(s)) - by_values.value(state_type(s))) < 1e-6;
    }

    // 3. Example 4.3: the gambler's problem, with a coin that lands heads 40% of the time, γ = 1
    std::cout << std::endl << "the gambler's problem, p_h = 0.4: the value of a capital, by value iteration" << std::endl;
    games::gamblers_problem gambler(0.4);
    auto bets = gambler.model();
    const std::vector<unsigned int> capitals = {1, 10, 25, 40, 50, 60, 75, 90, 99};
    std::cout << "    capital:     ";
    for (auto c : capitals) {
        std::cout << std::setw(7) << c;
    }
    std::cout << std::endl << std::setprecision(4);
    for (std::size_t k : {1u, 2u, 3u, 10u}) {
        solver_type dp(1.0, 1e-9);
        dp.max_sweeps = k;
        dp.value_iteration(bets);
        std::cout << "    sweep " << std::setw(2) << k << ":     ";
        for (auto c : capitals) {
            std::cout << std::setw(7) << dp.value(state_type(c));
        }
        std::cout << std::endl;
    }
    solver_type gambler_values(1.0, 1e-9);
    auto stakes = gambler_values.value_iteration(bets);
    std::cout << "    final:        ";
    for (auto c : capitals) {
        std::cout << std::setw(7) << gambler_values.value(state_type(c));
    }
    std::cout << std::endl << "    (" << gambler_values.sweeps() << " sweeps until no value changes by 1e-9)" << std::endl
              << "    the book's values: V(25) = 0.16, V(50) = 0.4, V(75) = 0.64" << std::endl;
    learnt = learnt && std::fabs(gambler_values.value(state_type(25)) - 0.16) < 1e-6 &&
                       std::fabs(gambler_values.value(state_type(50)) - 0.4) < 1e-6 &&
                       std::fabs(gambler_values.value(state_type(75)) - 0.64) < 1e-6;

    // the policy: many stakes are equally good, and the smallest of them gives Figure 4.3's
    std::cout << std::endl << "the stake of each capital (the smallest of the best ones), as in Figure 4.3:" << std::endl;
    for (int level = 50; level > 0; level -= 5) {
        std::cout << std::setw(6) << level << " |";
        for (unsigned int c = 1; c < 100; c++) {
            std::cout << (static_cast<int>(stakes.at(state_type(c)).trait()) > level - 5 ? '#' : ' ');
        }
        std::cout << std::endl;
    }
    std::cout << "       +" << std::string(99, '-') << std::endl
              << "        1" << std::string(23, ' ') << "25" << std::string(23, ' ') << "50"
              << std::string(23, ' ') << "75" << std::string(22, ' ') << "99   capital" << std::endl;

    // policy iteration finds the same values, though it may pick other stakes among the equally good ones
    solver_type gambler_policy(1.0, 1e-9);
    auto other_stakes = gambler_policy.policy_iteration(bets);
    double largest = 0;
    unsigned int differ = 0;
    for (unsigned int c = 1; c < 100; c++) {
        largest = std::max(largest, std::fabs(gambler_policy.value(state_type(c)) - gambler_values.value(state_type(c))));
        differ += !(other_stakes.at(state_type(c)) == stakes.at(state_type(c)));
    }
    std::cout << std::endl << std::scientific << std::setprecision(1)
              << "policy iteration (evaluations: " << gambler_policy.iterations() << ", sweeps: "
              << gambler_policy.sweeps() << "): the largest difference from value iteration's values is "
              << largest << "," << std::endl << "with other stakes at " << differ << " of the 99 capitals" << std::endl;
    learnt = learnt && largest < 1e-6;

    return learnt ? EXIT_SUCCESS : EXIT_FAILURE;
}
