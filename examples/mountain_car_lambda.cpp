/**
 * Mountain car (Gymnasium's MountainCar-v0 physics) learnt with true online Sarsa(λ), after every step
 * (Sutton & Barto, Example 10.1 and Section 12.7).
 *
 * The state (position, velocity) is cut into a 20 x 20 grid, as in `mountain_car.cpp`. The agent is
 * greedy, without random moves: every value starts at 0 and every step costs -1, so moves it has not
 * tried look best, and it explores by itself. There is no step limit, so an episode lasts until the
 * car reaches the flag; at first that takes thousands of steps.
 *
 * Each learner is trained in 20 runs of 20 episodes, learning after every step (`games::play_online`),
 * with α = 0.5 and γ = 1. λ = 0 is SARSA. The longer the traces, the more of the moves before each
 * step share in what it teaches. The first episode takes thousands of steps whatever λ is, but from
 * the second one on, the longer the traces, the fewer steps: with λ = 0.95 little more than half of SARSA's.
 * The program succeeds if λ = 0.9 and λ = 0.95 take fewer steps than SARSA, over all 20 episodes.
 *
 * Learning after every step matters here. Learning after each episode instead (`sarsa_lambda` after
 * `games::play`), the first episode is played with all the values at 0, by an agent that picks at
 * random between them, and nothing it does changes its mind until the episode is over: in a quick
 * test (replacing traces, λ = 0 to 0.9, α = 0.1 to 0.5, and a cap of 100000 steps), its first
 * 20 episodes took over 60000 steps each on average.
 *
 * @see games/mountain_car.hpp for the physics
 * @see docs/eligibility_traces.md for the algorithm
 */
#include "games/mountain_car.hpp"
#include "games/discretised.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// set shortcuts to the game, cut into a grid, and to its relearn types
using game_type = games::discretised<games::mountain_car>;
using types     = games::types<game_type>;

const double alpha = 0.5;
const unsigned int runs     = 20;
const unsigned int episodes = 20;

/// the steps one learner took to reach the flag, averaged over its runs
struct outcome
{
    /// in the first episode
    double first = 0;
    /// in episodes 2 to 20
    double later = 0;
    /// in all 20 episodes
    double all = 0;
};

/**
 * @brief train true online Sarsa(λ) in `runs` runs of `episodes` episodes, learning after every step
 * @param lambda is λ (0 is SARSA)
 * @return the steps per episode
 */
outcome train(double lambda)
{
    outcome result;
    for (unsigned int run = 1; run <= runs; run++) {
        // a new agent for each run: an empty policy, its own random numbers, and its own starts
        game_type game(games::mountain_car(run), {{-1.2, 0.6, 20}, {-0.07, 0.07, 20}});
        std::mt19937 prng(run);
        types::policy memory;
        relearn::true_online_sarsa_lambda<types::state, types::action> learner(alpha, 1.0, lambda);
        auto agent = [&](const unsigned int & s,
                         const std::vector<unsigned int> & legal) {
            return games::greedy<game_type>(memory, s, legal, prng);
        };
        for (unsigned int i = 1; i <= episodes; i++) {
            // no step limit: the episode lasts until the flag is reached
            auto r = games::play_online(game, agent, learner, memory, 1000000);
            double steps = static_cast<double>(r.steps);
            if (i == 1) {
                result.first += steps / runs;
            }
            else {
                result.later += steps / ((episodes - 1) * runs);
            }
            result.all += steps / (episodes * runs);
        }
    }
    return result;
}

int main()
{
    std::cout << "Mountain car, true online Sarsa(λ) learning after every step, α = " << alpha
              << ", " << runs << " runs of " << episodes << " episodes" << std::endl << std::endl;
    std::cout << "    the steps to reach the flag, per episode:" << std::endl;
    std::cout << "    λ                 first   2 to 20    all 20" << std::endl;
    std::cout << std::fixed << std::setprecision(0);
    const std::vector<double> lambdas = {0, 0.5, 0.9, 0.95};
    const std::vector<std::string> names = {"0 (SARSA)", "0.5", "0.9", "0.95"};
    std::vector<outcome> outcomes;
    for (std::size_t i = 0; i < lambdas.size(); i++) {
        outcomes.push_back(train(lambdas[i]));
        const auto & o = outcomes.back();
        std::cout << "    " << std::left << std::setw(12) << names[i] << std::right
                  << std::setw(10) << o.first << std::setw(10) << o.later << std::setw(10) << o.all << std::endl;
    }

    // longer traces must reach the flag in fewer steps than SARSA
    bool as_expected = (outcomes[2].all < outcomes[0].all) && (outcomes[3].all < outcomes[0].all);
    return as_expected ? EXIT_SUCCESS : EXIT_FAILURE;
}
