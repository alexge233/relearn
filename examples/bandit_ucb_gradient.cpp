/**
 * UCB and the gradient bandit on the 10-armed testbed (Sutton & Barto, Sections 2.7 and 2.8).
 *
 * Every run is a new bandit, learnt from scratch for 1000 pulls, and the results are
 * averaged over 2000 runs, as in the book:
 *
 * - Figure 2.4: the average reward of UCB (c = 2) against ε-greedy (ε = 0.1),
 *   both with sample averages. UCB earns more after the first steps, and has a
 *   spike at step 11, the first step after every arm has been tried once.
 * - Figure 2.5: how often the gradient bandit pulls the best arm, with α = 0.1 and
 *   0.4, with and without the baseline, on bandits whose means are around +4.
 *
 * @see games/bandit.hpp for the rules, and docs/bandits.md for the algorithms
 */
#include "games/bandit.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// set shortcuts to the game and to its relearn types
using game_type = games::bandit;
using types     = games::types<game_type>;

// the number of bandits, and of pulls of each one
const unsigned int runs  = 2000;
const unsigned int pulls = 1000;

/// @return the arms of @param game, as relearn actions
std::vector<types::action> arms_of(const game_type & game)
{
    std::vector<types::action> arms;
    for (auto a : game.actions(0)) {
        arms.push_back(types::action(a));
    }
    return arms;
}

/// @return the average reward at each pull 1 ... pulls of UCB with @param c (index 0 is unused)
std::vector<double> ucb_rewards(double c)
{
    std::vector<double> reward(pulls + 1, 0);
    for (unsigned int run = 0; run < runs; run++) {
        // a different bandit in every run
        game_type game(10, run);
        auto arms = arms_of(game);
        // the sample average of each arm, and UCB, which counts the pulls of each arm
        types::policy memory;
        relearn::ucb<types::state, types::action> learner(c, run + 1000);
        auto agent = [&](const unsigned int & s,
                         const std::vector<unsigned int> &) {
            return learner.choose(types::state(s), arms, memory).trait();
        };
        for (unsigned int i = 1; i <= pulls; i++) {
            // one pull is one episode
            auto r = games::play(game, agent, 1);
            reward[i] += r.total_reward / runs;
            learner(r.episode, memory);
        }
    }
    return reward;
}

/// @return the average reward at each pull 1 ... pulls of ε-greedy with @param epsilon (index 0 is unused)
std::vector<double> epsilon_greedy_rewards(double epsilon)
{
    std::vector<double> reward(pulls + 1, 0);
    for (unsigned int run = 0; run < runs; run++) {
        // the same bandits as UCB
        game_type game(10, run);
        std::mt19937 prng(run + 1000);
        // the sample average of each arm, as in examples/bandit.cpp
        types::policy memory;
        relearn::q_probabilistic<types::state, types::action> learner(1.0);
        auto agent = [&](const unsigned int & s,
                         const std::vector<unsigned int> & arms) {
            return games::epsilon_greedy<game_type>(memory, s, arms, epsilon, prng);
        };
        for (unsigned int i = 1; i <= pulls; i++) {
            auto r = games::play(game, agent, 1);
            reward[i] += r.total_reward / runs;
            learner(r.episode, memory, r.terminated);
        }
    }
    return reward;
}

/**
 * @brief the gradient bandit on bandits whose means are around +4
 * @param alpha is the step size α
 * @param baseline is true to use the average reward as the baseline
 * @return the share of runs pulling the best arm at each pull 1 ... pulls (index 0 is unused)
 */
std::vector<double> gradient_optimal(double alpha,
                                     bool baseline)
{
    std::vector<double> optimal(pulls + 1, 0);
    for (unsigned int run = 0; run < runs; run++) {
        // a different bandit in every run, with every mean shifted by +4
        game_type game(10, run, 4.0);
        auto arms = arms_of(game);
        // the preference H of each arm, and the gradient bandit, which keeps the average reward
        types::policy preferences;
        relearn::gradient_bandit<types::state, types::action> learner(alpha, baseline, run + 1000);
        auto agent = [&](const unsigned int & s,
                         const std::vector<unsigned int> &) {
            return learner.choose(types::state(s), arms, preferences).trait();
        };
        for (unsigned int i = 1; i <= pulls; i++) {
            auto r = games::play(game, agent, 1);
            if (r.episode[0].action.trait() == game.best_arm()) {
                optimal[i] += 1.0 / runs;
            }
            learner(r.episode, preferences, arms);
        }
    }
    return optimal;
}

/// @return the average of @param values over the pulls @param from ... @param to
double mean(const std::vector<double> & values,
            unsigned int from,
            unsigned int to)
{
    double sum = 0;
    for (unsigned int i = from; i <= to; i++) {
        sum += values[i];
    }
    return sum / (to - from + 1);
}

/// @brief print @param label, followed by spaces up to @param width characters
void print_label(const std::string & label,
                 std::size_t width)
{
    // count the characters, not the bytes: ε and α take two bytes each in UTF-8
    std::size_t characters = 0;
    for (char byte : label) {
        if ((static_cast<unsigned char>(byte) & 0xC0) != 0x80) {
            characters++;
        }
    }
    std::cout << "  " << label << std::string(width > characters ? width - characters : 0, ' ');
}

/// @brief print a row of @param values at the pulls @param at, after a @param label, as percentages if @param percent
void print_row(const std::string & label,
               const std::vector<double> & values,
               const std::vector<unsigned int> & at,
               bool percent)
{
    print_label(label, 28);
    for (unsigned int i : at) {
        if (percent) {
            std::cout << std::setw(7) << std::setprecision(1) << 100 * values[i] << "%";
        }
        else {
            std::cout << std::setw(8) << std::setprecision(3) << values[i];
        }
    }
}

int main()
{
    std::cout << std::fixed;
    std::cout << "the 10-armed testbed: " << runs << " bandits, " << pulls << " pulls each" << std::endl;

    // Figure 2.4: UCB against ε-greedy
    const auto ucb = ucb_rewards(2.0);
    const auto greedy = epsilon_greedy_rewards(0.1);
    const std::vector<unsigned int> reward_at = {1, 10, 11, 12, 100, 1000};
    std::cout << std::endl << "Figure 2.4: the average reward at pull" << std::endl;
    print_label("", 28);
    for (unsigned int i : reward_at) {
        std::cout << std::setw(8) << i;
    }
    std::cout << "   pulls 101 to 1000" << std::endl;
    print_row("UCB, c = 2", ucb, reward_at, false);
    std::cout << std::setw(10) << std::setprecision(3) << mean(ucb, 101, pulls) << std::endl;
    print_row("ε-greedy, ε = 0.1", greedy, reward_at, false);
    std::cout << std::setw(10) << std::setprecision(3) << mean(greedy, 101, pulls) << std::endl;

    // Figure 2.5: the gradient bandit, with and without the baseline
    const auto with_01    = gradient_optimal(0.1, true);
    const auto with_04    = gradient_optimal(0.4, true);
    const auto without_01 = gradient_optimal(0.1, false);
    const auto without_04 = gradient_optimal(0.4, false);
    const std::vector<unsigned int> optimal_at = {10, 100, 250, 500, 1000};
    std::cout << std::endl << "Figure 2.5: the share of runs pulling the best arm (means around +4), at pull" << std::endl;
    print_label("", 28);
    for (unsigned int i : optimal_at) {
        std::cout << std::setw(8) << i;
    }
    std::cout << "   pulls 901 to 1000" << std::endl;
    const std::vector<std::pair<std::string, const std::vector<double> *>> rows = {
        {"α = 0.1, with baseline", &with_01},
        {"α = 0.4, with baseline", &with_04},
        {"α = 0.1, without baseline", &without_01},
        {"α = 0.4, without baseline", &without_04}};
    for (const auto & row : rows) {
        print_row(row.first, *row.second, optimal_at, true);
        std::cout << std::setw(9) << std::setprecision(1) << 100 * mean(*row.second, 901, pulls) << "%" << std::endl;
    }

    // UCB earns more than ε-greedy after the first steps, with a spike at step 11
    bool learnt = mean(ucb, 101, pulls) > mean(greedy, 101, pulls);
    learnt = learnt && ucb[11] > ucb[10] && ucb[11] > ucb[12];
    // and the baseline makes the gradient bandit far better, with either step size
    learnt = learnt && mean(with_01, 901, pulls) > mean(without_01, 901, pulls) + 0.2;
    learnt = learnt && mean(with_04, 901, pulls) > mean(without_04, 901, pulls) + 0.2;
    return learnt ? EXIT_SUCCESS : EXIT_FAILURE;
}
