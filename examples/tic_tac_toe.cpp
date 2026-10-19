/**
 * Tic-tac-toe (Sutton & Barto, Section 1.5) learnt with `q_learning`.
 *
 * Two agents learn by playing 20000 games, one against a random opponent and
 * one against a perfect (minimax) opponent, and both are then tested against
 * each opponent. The perfect opponent always plays the same way, so the agent
 * trained against it only ever sees one line of play: it learns to draw that
 * line, but has never met the positions a random player creates.
 *
 * @see games/tic_tac_toe.hpp for the rules
 */
#include "games/tic_tac_toe.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

// set shortcuts to the game and to its relearn types
using game_type = games::tic_tac_toe;
using types     = games::types<game_type>;

/**
 * @brief play the greedy policy @param memory in @param games_played games of @param game
 * @param prng breaks ties between moves of equal value
 * @return the number of {wins, draws, losses}
 */
std::vector<int> results(game_type & game,
                         const types::policy & memory,
                         int games_played,
                         std::mt19937 & prng)
{
    // the greedy agent: always the best known move
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::greedy<game_type>(memory, s, legal, prng);
    };
    std::vector<int> wdl(3, 0);
    for (int i = 0; i < games_played; i++) {
        auto r = games::play(game, agent, 9);
        // the reward of a game is +1 for a win, 0 for a draw, -1 for a loss
        if (r.total_reward > 0) {
            wdl[0]++;
        }
        else if (r.total_reward == 0) {
            wdl[1]++;
        }
        else {
            wdl[2]++;
        }
    }
    return wdl;
}

int main()
{
    std::mt19937 prng(2);
    bool ok = true;
    // train one agent against each opponent
    for (auto trainer : {game_type::random, game_type::perfect}) {
        game_type game(trainer, 1);
        types::policy memory;
        relearn::q_learning<types::state, types::action> learner;
        learner.alpha = 0.3;
        learner.gamma = 1.0;

        // the ε-greedy agent, with ε = 0.1
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        for (int i = 0; i < 20000; i++) {
            auto r = games::play(game, agent, 9);
            learner(r.episode, memory, r.terminated);
        }

        // test it against both opponents: 10000 games against the random one,
        // and one game against the perfect one, which always plays the same way
        game_type random_opponent(game_type::random, 7);
        game_type perfect_opponent(game_type::perfect);
        auto vs_random  = results(random_opponent, memory, 10000, prng);
        auto vs_perfect = results(perfect_opponent, memory, 1, prng);

        // print the results, as percentages of the 10000 games
        std::cout << "trained against a " << (trainer == game_type::random ? "random" : "perfect")
                  << " opponent:" << std::endl;
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "    vs random : " << std::setw(5) << vs_random[0] / 100.0 << "% wins  "
                  << std::setw(5) << vs_random[1] / 100.0 << "% draws  "
                  << std::setw(5) << vs_random[2] / 100.0 << "% losses" << std::endl;
        std::cout << "    vs perfect: ";
        if (vs_perfect[1]) {
            std::cout << "draw (the best possible)" << std::endl;
        }
        else if (vs_perfect[2]) {
            std::cout << "loss" << std::endl;
        }
        else {
            std::cout << "win" << std::endl;
        }

        // the agent trained against the random opponent must win over 90% of its games,
        // and the one trained against the perfect opponent must draw it
        if (trainer == game_type::random) {
            ok = ok && vs_random[0] > 9000;
        }
        else {
            ok = ok && vs_perfect[1] == 1;
        }
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
