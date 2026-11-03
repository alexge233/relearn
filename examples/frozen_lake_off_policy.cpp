/**
 * FrozenLake (slippery 4x4, γ = 0.95) learnt off-policy, from an agent that only ever moves at random
 * (Sutton & Barto, Sections 12.9 and 12.10).
 *
 * 1. Control: the optimal values q*, learnt with a greedy π from random play (b = 1/4 for every move).
 *    The RMS error of Q against q* (worked out by value iteration on the game's model), after 100, 1000
 *    and 10000 episodes, averaged over 20 runs, for Expected SARSA (λ = 0), Watkins's Q(λ), Tree-Backup(λ)
 *    and off-policy Expected Sarsa(λ) with importance sampling. Tree-Backup(λ)'s traces only ever shrink;
 *    importance sampling's grow by ρ = 4 at every greedy move, and with λ = 0.9 its values blow up in about
 *    half the runs.
 *    With a uniform b, Tree-Backup(λ) is importance sampling with λ / 4: Tree-Backup(1) and importance
 *    sampling with λ = 0.25 learn the very same values.
 * 2. Prediction: the state values of a policy that takes the optimal move with 0.85 and each other one with
 *    0.05, learnt with off-policy TD(λ) from the same random play. With λ = 0.9, γλρ = 2.9 at every move π
 *    prefers, and in about half the runs the values blow up too.
 *
 * The program succeeds if Tree-Backup(0.9) ends within 0.07 of q*, if Tree-Backup(1) and importance sampling
 * with λ = 0.25 learn the same values to within 1e-12, if importance sampling with λ = 0.9 ends further than 1
 * away on average, and if off-policy TD(λ) ends within 0.08 of the true values with λ = 0 and 0.1 with 0.5, and
 * further than 1 with 0.9.
 *
 * @see games/frozen_lake.hpp for the rules
 * @see docs/off_policy_traces.md for the algorithms
 */
#include "games/frozen_lake.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// set shortcuts to the game and to its relearn types
using game_type = games::frozen_lake;
using types     = games::types<game_type>;

/// @return true for a hole or the goal of @param lake, where nothing is learnt
bool ends(const game_type & lake,
          unsigned int s)
{
    return lake.tile(s) == 'H' || lake.tile(s) == 'G';
}

/// @return the RMS of @param errors, or 1000 if it is not a number or larger
double rms(const std::vector<double> & errors)
{
    double sum = 0;
    for (auto e : errors) {
        sum += e * e;
    }
    double r = std::sqrt(sum / static_cast<double>(errors.size()));
    return std::isfinite(r) ? std::min(r, 1000.0) : 1000.0;
}

/// @return @param name padded with spaces to @param width characters (λ is two bytes, but one character)
std::string padded(const std::string & name,
                   std::size_t width)
{
    std::size_t characters = 0;
    for (unsigned char c : name) {
        if ((c & 0xC0) != 0x80) {
            characters++;
        }
    }
    return name + std::string(width > characters ? width - characters : 0, ' ');
}

/// @brief print a row of errors, @param name first
void print_row(const std::string & name,
               const std::vector<double> & errors)
{
    std::cout << "    " << padded(name, 34);
    for (auto e : errors) {
        if (e >= 1000) {
            std::cout << std::setw(11) << "blown up";
        }
        else {
            std::cout << std::setw(11) << std::setprecision(4) << e;
        }
    }
    std::cout << std::endl;
}

int main()
{
    std::cout << std::fixed;
    const double gamma = 0.95;
    const unsigned int runs = 20;
    const std::vector<unsigned int> marks = {100, 1000, 10000};
    game_type probe(game_type::map_4x4(), true);
    auto model = probe.model();
    relearn::dynamic_programming<types::state, types::action> optimal;
    optimal.gamma = gamma;
    optimal.theta = 1e-12;
    optimal.value_iteration(model);
    auto q_star = [&](unsigned int s, unsigned int a) {
        return optimal.q_value(model, types::state(s), types::action(a));
    };
    // the random agent, and b = 1/4 for every move
    auto b = [](const types::state &, const types::action &) { return 0.25; };

    // 1. control: q*, with a greedy π
    std::cout << "FrozenLake, q* learnt from random play: the RMS error of Q (20 runs)" << std::endl;
    std::cout << "    " << padded("episodes:", 34);
    for (auto m : marks) {
        std::cout << std::setw(11) << m;
    }
    std::cout << std::endl;
    const std::vector<std::string> names = {"Expected SARSA (λ = 0)", "Watkins's Q(λ), λ = 0.9", "Tree-Backup(λ), λ = 0.5",
                                            "Tree-Backup(λ), λ = 0.9", "Tree-Backup(λ), λ = 1",
                                            "importance sampling, λ = 0.25", "importance sampling, λ = 0.9"};
    std::vector<std::vector<double>> control(names.size(), std::vector<double>(marks.size(), 0));
    double same = 0;
    for (unsigned int run = 0; run < runs; run++) {
        std::vector<types::policy> learnt(names.size());
        for (std::size_t which = 0; which < names.size(); which++) {
            // the same games, and the same random moves, for every learner
            game_type lake(game_type::map_4x4(), true, run);
            std::mt19937 prng(run + 500);
            std::uniform_int_distribution<unsigned int> pick(0, 3);
            auto random_player = [&](const unsigned int &, const std::vector<unsigned int> &) { return pick(prng); };
            relearn::tree_backup_lambda<types::state, types::action> tree(0.1, gamma, which == 0 ? 0.0 : (which == 2 ? 0.5 : (which == 3 ? 0.9 : 1.0)));
            relearn::watkins_q_lambda<types::state, types::action> watkins(0.1, gamma, 0.9);
            relearn::off_policy_expected_sarsa_lambda<types::state, types::action> sampled(0.1, gamma, which == 5 ? 0.25 : 0.9, nullptr, b);
            std::size_t m = 0;
            for (unsigned int e = 1; e <= marks.back(); e++) {
                auto r = games::play(lake, random_player, lake.step_limit());
                if (which == 1) {
                    watkins(r.episode, learnt[which], r.terminated);
                }
                else if (which >= 5) {
                    sampled(r.episode, learnt[which], r.terminated);
                }
                else {
                    tree(r.episode, learnt[which], r.terminated);
                }
                if (e == marks[m]) {
                    std::vector<double> errors;
                    for (unsigned int s = 0; s < 16; s++) {
                        for (unsigned int a = 0; a < 4 && !ends(probe, s); a++) {
                            errors.push_back(learnt[which].value(types::state(s), types::action(a)) - q_star(s, a));
                        }
                    }
                    control[which][m] += rms(errors) / runs;
                    m++;
                }
            }
        }
        // Tree-Backup(1) and importance sampling with λ = 1/4 learn the very same values
        for (unsigned int s = 0; s < 16; s++) {
            for (unsigned int a = 0; a < 4; a++) {
                same = std::max(same, std::fabs(learnt[4].value(types::state(s), types::action(a)) -
                                                learnt[5].value(types::state(s), types::action(a))));
            }
        }
    }
    for (std::size_t which = 0; which < names.size(); which++) {
        print_row(names[which], control[which]);
    }
    std::cout << "    Tree-Backup(1) and importance sampling with λ = 0.25: the largest difference of their values is "
              << std::scientific << std::setprecision(1) << same << std::fixed << std::endl;

    // 2. prediction: v_π, with π the optimal move with 0.85 and each other one with 0.05
    auto best = [&](unsigned int s) {
        unsigned int a = 0;
        for (unsigned int k = 1; k < 4; k++) {
            if (q_star(s, k) > q_star(s, a) + 1e-12) {
                a = k;
            }
        }
        return a;
    };
    auto pi = [&](const types::state & s, const types::action & a) { return a.trait() == best(s.trait()) ? 0.85 : 0.05; };
    relearn::dynamic_programming<types::state, types::action> exact;
    exact.gamma = gamma;
    exact.theta = 1e-12;
    exact.evaluate(model, pi);
    std::cout << std::endl << "FrozenLake, the values of a policy learnt from random play with off-policy TD(λ): the RMS error of V (20 runs)" << std::endl;
    std::cout << "    " << padded("episodes:", 34);
    for (auto m : marks) {
        std::cout << std::setw(11) << m;
    }
    std::cout << std::endl;
    const std::vector<double> lambdas = {0, 0.5, 0.9};
    std::vector<std::vector<double>> prediction(lambdas.size(), std::vector<double>(marks.size(), 0));
    for (std::size_t j = 0; j < lambdas.size(); j++) {
        for (unsigned int run = 0; run < runs; run++) {
            game_type lake(game_type::map_4x4(), true, run);
            std::mt19937 prng(run + 500);
            std::uniform_int_distribution<unsigned int> pick(0, 3);
            auto random_player = [&](const unsigned int &, const std::vector<unsigned int> &) { return pick(prng); };
            relearn::off_policy_td_lambda<types::state, types::action> td(0.02, gamma, lambdas[j], pi, b);
            std::size_t m = 0;
            for (unsigned int e = 1; e <= marks.back(); e++) {
                auto r = games::play(lake, random_player, lake.step_limit());
                td(r.episode, r.terminated);
                if (e == marks[m]) {
                    std::vector<double> errors;
                    for (unsigned int s = 0; s < 16; s++) {
                        if (!ends(probe, s)) {
                            errors.push_back(td.value(types::state(s)) - exact.value(types::state(s)));
                        }
                    }
                    prediction[j][m] += rms(errors) / runs;
                    m++;
                }
            }
        }
        print_row("off-policy TD(λ), λ = " + std::to_string(lambdas[j]).substr(0, 3), prediction[j]);
    }

    const bool tree_learns = control[3].back() < 0.07;
    const bool identical = same < 1e-12;
    const bool sampling_blows_up = control[6].back() > 1;
    const bool td_learns = prediction[0].back() < 0.08 && prediction[1].back() < 0.1 && prediction[2].back() > 1;
    return (tree_learns && identical && sampling_blows_up && td_learns) ? EXIT_SUCCESS : EXIT_FAILURE;
}
