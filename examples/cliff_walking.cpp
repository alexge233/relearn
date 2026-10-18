/**
 * Cliff Walking (Sutton & Barto, Example 6.6) learnt with `q_learning`.
 *
 * Q-learning learns the shortest path, right along the edge of the cliff
 * (13 moves). While it explores with ε-greedy it still falls off now and then,
 * so its return during training stays below that of the safer path SARSA would
 * learn: the classic on-policy vs off-policy example.
 *
 * @see games/cliff_walking.hpp for the rules
 */
#include "games/cliff_walking.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

using game_type = games::cliff_walking;
using types     = games::types<game_type>;

/// draw the grid with the cells visited by the greedy policy
void draw(const types::policy & memory, std::mt19937 & prng)
{
    std::vector<std::string> rows(game_type::height, std::string(game_type::width, '.'));
    for (int x = 1; x < game_type::width - 1; x++) {
        rows[game_type::height - 1][x] = 'C';
    }
    game_type game;
    auto s = game.reset();
    for (int step = 0; step < 100; step++) {
        rows[s.y][s.x] = '*';
        auto out = game.step(games::greedy<game_type>(memory, s, game.actions(s), prng));
        s = out.next;
        if (out.terminated) break;
    }
    rows[game_type::start().y][game_type::start().x] = 'S';
    rows[game_type::goal().y][game_type::goal().x]   = 'G';
    for (const auto & row : rows) {
        std::printf("    %s\n", row.c_str());
    }
}

int main()
{
    const unsigned int episodes = 1000;
    const double epsilon = 0.1;
    game_type game;
    std::mt19937 prng(1);
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.5;
    learner.gamma = 1.0;

    double sum = 0;
    for (unsigned int i = 1; i <= episodes; i++) {
        auto r = games::play(game,
                             [&](const game_type::state_type & s, const std::vector<unsigned int> & legal) {
                                 return games::epsilon_greedy<game_type>(memory, s, legal, epsilon, prng);
                             }, 1000);
        learner(r.episode, memory, r.terminated);
        sum += r.total_reward;
        if (i % 100 == 0) {
            std::printf("episodes %4u-%4u: average return while exploring %7.1f\n", i - 99, i, sum / 100);
            sum = 0;
        }
    }
    double greedy = games::evaluate(game, memory, 1, 100, prng);
    std::printf("\ngreedy policy: return %.0f (the shortest path is 13 moves, -13)\n", greedy);
    draw(memory, prng);
    return greedy == -13 ? EXIT_SUCCESS : EXIT_FAILURE;
}
