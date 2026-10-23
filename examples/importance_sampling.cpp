/**
 * Off-policy Monte Carlo prediction: learning the values of one policy, π, from episodes played by another, b,
 * with importance sampling (Sutton & Barto, Sections 5.5, 5.8 and 5.9), in three parts.
 *
 * 1. Example 5.4 and Figure 5.3: one blackjack state, the dealer showing a deuce and the player on 13 with a
 *    usable Ace. π sticks only on 20 or 21, b picks stick or hit at random, γ = 1, and v_π(s) = -0.27726.
 *    The mean squared error of V(s), over 100 runs, after 1 to 10000 games: weighted importance sampling is far
 *    better at first, and both get there.
 * 2. Example 5.5: one state, where π always goes left and b goes left or right at random. v_π(s) = 1, and
 *    ordinary importance sampling has an infinite variance: after 100000 episodes its estimates, in 10 runs,
 *    are still all over the place, whereas weighted importance sampling has 1 exactly from the first episode
 *    π could have played (first-visit, as in the book).
 * 3. Long episodes with a reward at every step (`games::fixed_length`, 20 steps, γ = 0.5): ordinary importance
 *    sampling weights every return by all 20 ratios of the episode, even those of the actions after its rewards.
 *    Discounting-aware (Section 5.8) and per-decision (Section 5.9) importance sampling don't, and after 1000
 *    episodes their errors are thousands of times smaller (2000 to 8000 times with GCC, 1200 to 5500 with libc++).
 *
 * The program succeeds if weighted importance sampling has the smaller error after 1 game in part 1 and
 * both are below 0.002 after 10000, if in part 2 weighted has exactly 1 in every run and ordinary's 10 runs are
 * at least 0.3 apart, and if in part 3 discounting-aware and per-decision have less than 1/100 of ordinary's error.
 *
 * @see games/blackjack.hpp, games/infinite_variance.hpp and games/fixed_length.hpp for the rules
 * @see docs/monte_carlo.md for the algorithms
 */
#include "games/blackjack.hpp"
#include "games/infinite_variance.hpp"
#include "games/fixed_length.hpp"
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using sampling_type = relearn::importance_sampling;

/// @return the name of the importance sampling @param s
std::string name_of(sampling_type s)
{
    switch (s) {
        case sampling_type::ordinary: return "ordinary";
        case sampling_type::weighted: return "weighted";
        case sampling_type::discounting_aware_ordinary: return "discounting-aware ordinary";
        case sampling_type::discounting_aware_weighted: return "discounting-aware weighted";
        default: return "per-decision";
    }
}

/**
 * @brief part 1, Figure 5.3: the mean squared error of V(s) for Example 5.4's blackjack state
 * @param kind is the importance sampling
 * @param marks are the numbers of games after which the error is measured
 * @return the error after each of them, over 100 runs
 */
std::vector<double> figure_5_3(sampling_type kind,
                               const std::vector<unsigned int> & marks)
{
    using game_type = games::blackjack;
    using types = games::types<game_type>;
    const double truth = -0.27726;
    // π sticks only on 20 or 21, b picks at random
    auto pi = [](const types::state & s, const types::action & a) {
        const bool stick = s.trait().player_sum >= 20;
        return (a.trait() == game_type::stick) == stick ? 1.0 : 0.0;
    };
    auto b = [](const types::state &, const types::action &) { return 0.5; };
    // the state where every game starts
    game_type::state_type start;
    start.player_sum = 13;
    start.usable_ace = true;
    start.dealer_showing = 2;
    std::vector<double> error(marks.size(), 0);
    for (unsigned int run = 0; run < 100; run++) {
        game_type game(run);
        game.fixed_start(13, true, 2);
        std::mt19937 prng(run + 500);
        std::uniform_int_distribution<unsigned int> coin(0, 1);
        auto random_player = [&](const game_type::state_type &, const std::vector<unsigned int> &) { return coin(prng); };
        relearn::off_policy_monte_carlo_prediction<types::state, types::action> learner(1.0, pi, b, kind);
        types::policy q;
        std::size_t m = 0;
        for (unsigned int e = 1; e <= marks.back(); e++) {
            auto r = games::play(game, random_player, 100);
            learner(r.episode, q, r.terminated);
            if (e == marks[m]) {
                const double v = learner.value(types::state(start));
                error[m] += (v - truth) * (v - truth) / 100;
                m++;
            }
        }
    }
    return error;
}

/**
 * @brief part 2, Example 5.5: the estimates of v_π(s) = 1, in 10 runs
 * @param kind is the importance sampling
 * @param marks are the numbers of episodes after which the estimates are printed
 * @return the estimate of each run after the last of them
 */
std::vector<double> example_5_5(sampling_type kind,
                                const std::vector<unsigned int> & marks)
{
    using types = games::types<games::infinite_variance>;
    auto pi = [](const types::state &, const types::action & a) { return a.trait() == games::infinite_variance::left ? 1.0 : 0.0; };
    auto b = [](const types::state &, const types::action &) { return 0.5; };
    std::vector<double> last;
    std::cout << "    " << std::left << std::setw(10) << name_of(kind) << std::right;
    for (auto mark : marks) {
        std::cout << std::setw(9) << mark;
    }
    std::cout << "  episodes" << std::endl;
    for (unsigned int run = 0; run < 10; run++) {
        games::infinite_variance task(run);
        std::mt19937 prng(100 + run);
        std::uniform_int_distribution<unsigned int> coin(0, 1);
        auto random_player = [&](const unsigned int &, const std::vector<unsigned int> &) { return coin(prng); };
        relearn::off_policy_monte_carlo_prediction<types::state, types::action> learner(1.0, pi, b, kind, true);
        types::policy q;
        std::size_t m = 0;
        std::cout << "    run " << std::setw(2) << run + 1 << "    ";
        for (unsigned int e = 1; e <= marks.back(); e++) {
            auto r = games::play(task, random_player, 100000);
            learner(r.episode, q, r.terminated);
            if (e == marks[m]) {
                std::cout << std::setw(9) << learner.value(types::state(0));
                m++;
            }
        }
        std::cout << std::endl;
        last.push_back(learner.value(types::state(0)));
    }
    return last;
}

/**
 * @brief part 3: the mean squared error of V(0) on long episodes with a reward at every step, over 100 runs
 * @param kind is the importance sampling
 * @param marks are the numbers of episodes after which the error is measured
 * @return the error after each of them
 */
std::vector<double> long_episodes(sampling_type kind,
                                  const std::vector<unsigned int> & marks)
{
    using types = games::types<games::fixed_length>;
    games::fixed_length task(20);
    const double gamma = 0.5;
    const double v = task.value(0, 0.9, gamma);
    // π takes action 0 (worth 1) with 0.9, b with 0.5
    auto pi = [](const types::state &, const types::action & a) { return a.trait() == 0 ? 0.9 : 0.1; };
    auto b = [](const types::state &, const types::action &) { return 0.5; };
    std::vector<double> error(marks.size(), 0);
    for (unsigned int run = 0; run < 100; run++) {
        std::mt19937 prng(1000 + run);
        std::uniform_int_distribution<unsigned int> coin(0, 1);
        auto random_player = [&](const unsigned int &, const std::vector<unsigned int> &) { return coin(prng); };
        relearn::off_policy_monte_carlo_prediction<types::state, types::action> learner(gamma, pi, b, kind);
        types::policy q;
        std::size_t m = 0;
        for (unsigned int e = 1; e <= marks.back(); e++) {
            auto r = games::play(task, random_player, 100);
            learner(r.episode, q, r.terminated);
            if (e == marks[m]) {
                const double estimate = learner.value(types::state(0));
                error[m] += (estimate - v) * (estimate - v) / 100;
                m++;
            }
        }
    }
    return error;
}

/// @brief print a row of errors, @param name first
void print_row(const std::string & name,
               const std::vector<double> & errors)
{
    std::cout << "    " << std::left << std::setw(28) << name << std::right;
    for (auto e : errors) {
        std::cout << std::setw(11) << std::setprecision(e < 0.01 ? 6 : 4) << e;
    }
    std::cout << std::endl;
}

int main()
{
    std::cout << std::fixed;

    // 1. Example 5.4 and Figure 5.3
    std::cout << "Example 5.4: a blackjack state, v_π(s) = -0.27726, learnt from random play (Figure 5.3)" << std::endl;
    std::cout << "    the mean squared error of V(s), over 100 runs, after" << std::endl;
    const std::vector<unsigned int> games_played = {1, 10, 100, 1000, 10000};
    std::cout << "    " << std::setw(28) << "games:";
    for (auto g : games_played) {
        std::cout << std::setw(11) << g;
    }
    std::cout << std::endl;
    auto ordinary = figure_5_3(sampling_type::ordinary, games_played);
    auto weighted = figure_5_3(sampling_type::weighted, games_played);
    print_row("ordinary", ordinary);
    print_row("weighted", weighted);

    // 2. Example 5.5
    std::cout << std::endl << "Example 5.5: one state, v_π(s) = 1, ordinary importance sampling has an infinite variance (first-visit)" << std::endl;
    std::cout << std::setprecision(4);
    const std::vector<unsigned int> episodes = {10, 100, 1000, 10000, 100000};
    auto loose = example_5_5(sampling_type::ordinary, episodes);
    auto exact = example_5_5(sampling_type::weighted, episodes);

    // 3. long episodes with a reward at every step
    std::cout << std::endl << "Long episodes: 20 steps, a reward at every step, γ = 0.5 (Sections 5.8 and 5.9)" << std::endl;
    std::cout << "    the mean squared error of V(0), over 100 runs, after" << std::endl;
    const std::vector<unsigned int> long_marks = {10, 100, 1000};
    std::cout << "    " << std::setw(28) << "episodes:";
    for (auto e : long_marks) {
        std::cout << std::setw(11) << e;
    }
    std::cout << std::endl;
    std::vector<std::vector<double>> errors;
    for (auto kind : {sampling_type::ordinary, sampling_type::weighted, sampling_type::discounting_aware_ordinary,
                      sampling_type::discounting_aware_weighted, sampling_type::per_decision}) {
        errors.push_back(long_episodes(kind, long_marks));
        print_row(name_of(kind), errors.back());
    }

    const bool weighted_better = weighted[0] < ordinary[0] && ordinary.back() < 0.002 && weighted.back() < 0.002;
    const bool one_exactly = std::all_of(exact.begin(), exact.end(), [](double v) { return v == 1.0; });
    const bool spread = *std::max_element(loose.begin(), loose.end()) - *std::min_element(loose.begin(), loose.end()) > 0.3;
    const bool aware = errors[2].back() < 0.01 * errors[0].back() && errors[4].back() < 0.01 * errors[0].back();
    return (weighted_better && one_exactly && spread && aware) ? EXIT_SUCCESS : EXIT_FAILURE;
}
