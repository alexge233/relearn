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
#include <cstdio>
#include <cstdlib>

using game_type = games::tic_tac_toe;
using types     = games::types<game_type>;

/// @return {wins, draws, losses} of the greedy policy in @param games_played games
std::vector<int> results(game_type & game, const types::policy & memory, int games_played, std::mt19937 & prng)
{
    std::vector<int> wdl(3, 0);
    for (int i = 0; i < games_played; i++) {
        auto r = games::play(game,
                             [&](const game_type::state_type & s, const std::vector<unsigned int> & legal) {
                                 return games::greedy<game_type>(memory, s, legal, prng);
                             }, 9);
        wdl[r.total_reward > 0 ? 0 : (r.total_reward == 0 ? 1 : 2)]++;
    }
    return wdl;
}

int main()
{
    std::mt19937 prng(2);
    bool ok = true;
    for (auto trainer : {game_type::random, game_type::perfect}) {
        game_type game(trainer, 1);
        types::policy memory;
        relearn::q_learning<types::state, types::action> learner;
        learner.alpha = 0.3;
        learner.gamma = 1.0;
        for (int i = 0; i < 20000; i++) {
            auto r = games::play(game,
                                 [&](const game_type::state_type & s, const std::vector<unsigned int> & legal) {
                                     return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
                                 }, 9);
            learner(r.episode, memory, r.terminated);
        }
        game_type random_opponent(game_type::random, 7);
        game_type perfect_opponent(game_type::perfect);
        auto vs_random  = results(random_opponent, memory, 10000, prng);
        auto vs_perfect = results(perfect_opponent, memory, 1, prng);
        std::printf("trained against a %s opponent:\n", trainer == game_type::random ? "random" : "perfect");
        std::printf("    vs random : %5.1f%% wins  %5.1f%% draws  %5.1f%% losses\n",
                    vs_random[0] / 100.0, vs_random[1] / 100.0, vs_random[2] / 100.0);
        std::printf("    vs perfect: %s\n", vs_perfect[1] ? "draw (the best possible)" : (vs_perfect[2] ? "loss" : "win"));
        if (trainer == game_type::random) ok = ok && vs_random[0] > 9000;
        else ok = ok && vs_perfect[1] == 1;
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
