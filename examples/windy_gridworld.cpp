/**
 * Windy Gridworld (Sutton & Barto, Example 6.5 and Exercise 6.10).
 *
 * 1. Steady wind: `q_learning` learns the shortest path (15 moves).
 * 2. Random wind: the same move can end in different cells. `q_learning` with a
 *    fixed learning rate keeps chasing the latest outcome, while
 *    `q_probabilistic` averages the outcomes it has seen, and learns the
 *    policy with the fewest expected moves.
 *
 * @see games/windy_gridworld.hpp for the rules
 */
#include "games/windy_gridworld.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

using game_type = games::windy_gridworld;
using types     = games::types<game_type>;

/// train @param learner for @param episodes ε-greedy episodes
template <class learner_type>
types::policy train(game_type & game, learner_type & learner, unsigned int episodes, std::mt19937 & prng)
{
    types::policy memory;
    for (unsigned int i = 0; i < episodes; i++) {
        auto r = games::play(game,
                             [&](const game_type::state_type & s, const std::vector<unsigned int> & legal) {
                                 return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
                             }, 1000);
        learner(r.episode, memory, r.terminated);
    }
    return memory;
}

/// draw the cells visited by the greedy policy
void draw(game_type & game, const types::policy & memory, std::mt19937 & prng)
{
    std::vector<std::string> rows(game_type::height, std::string(game_type::width, '.'));
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
    std::printf("    ");
    for (int x = 0; x < game_type::width; x++) std::printf("%d", game_type::wind(x));
    std::printf("   <- wind\n");
}

int main()
{
    std::mt19937 prng(1);

    // 1. steady wind
    game_type steady;
    relearn::q_learning<types::state, types::action> q_steady;
    q_steady.alpha = 0.5;
    q_steady.gamma = 1.0;
    auto memory = train(steady, q_steady, 500, prng);
    double moves = -games::evaluate(steady, memory, 1, 100, prng);
    std::printf("steady wind, q_learning: %.0f moves (the shortest path is 15)\n", moves);
    draw(steady, memory, prng);

    // 2. random wind
    game_type random_wind(true, 2);
    relearn::q_learning<types::state, types::action> q_random;
    q_random.alpha = 0.5;
    q_random.gamma = 1.0;
    relearn::q_probabilistic<types::state, types::action> p_random(1.0);
    auto memory_q = train(random_wind, q_random, 2000, prng);
    auto memory_p = train(random_wind, p_random, 2000, prng);
    // a policy that wanders forever is stopped at 1000 moves
    double moves_q = -games::evaluate(random_wind, memory_q, 1000, 1000, prng);
    double moves_p = -games::evaluate(random_wind, memory_p, 1000, 1000, prng);
    std::printf("\nrandom wind, average moves over 1000 greedy episodes:\n");
    std::printf("    q_learning      %7.1f\n", moves_q);
    std::printf("    q_probabilistic %7.1f   (the best policy needs 20.6 on average)\n", moves_p);
    return moves == 15 ? EXIT_SUCCESS : EXIT_FAILURE;
}
