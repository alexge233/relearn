/**
 * Windy Gridworld (Sutton & Barto, Example 6.5 and Exercise 6.10).
 *
 * 1. Steady wind: `q_learning` learns the shortest path (15 moves).
 * 2. Random wind: the same move can end in different cells.
 *    - `q_learning` with a large constant learning rate (α = 0.5) keeps chasing
 *      the latest outcome, and does not settle on a good policy.
 *    - `q_learning` with a learning rate that decays for each state/action pair,
 *      α = 1/n^0.6 for its n-th update, averages the outcomes out, and learns
 *      the policy with the fewest expected moves.
 *    - `q_probabilistic` averages the outcomes it has seen by design, and learns
 *      that policy too.
 *
 * @see docs/schedules.md for learning rates that decay
 *
 * @see games/windy_gridworld.hpp for the rules
 */
#include "games/windy_gridworld.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

// set shortcuts to the game and to its relearn types
using game_type = games::windy_gridworld;
using types     = games::types<game_type>;

/**
 * @brief train @param learner for @param episodes ε-greedy episodes of @param game
 * @param prng draws the random choices of the agent
 * @return the learnt policy
 */
template <class learner_type>
types::policy train(game_type & game,
                    learner_type & learner,
                    unsigned int episodes,
                    std::mt19937 & prng)
{
    types::policy memory;
    // the ε-greedy agent: explores 10% of the time, otherwise takes the best known move
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
    };
    // play and learn after every episode
    for (unsigned int i = 0; i < episodes; i++) {
        auto r = games::play(game, agent, 1000);
        learner(r.episode, memory, r.terminated);
    }
    return memory;
}

/**
 * @brief draw the grid, marking with `*` the cells the greedy policy visits
 * @param game is the game to play
 * @param memory is the learnt policy
 * @param prng breaks ties between moves of equal value
 */
void draw(game_type & game,
          const types::policy & memory,
          std::mt19937 & prng)
{
    std::vector<std::string> rows(game_type::height, std::string(game_type::width, '.'));
    // follow the greedy policy from the start, marking each cell visited
    auto s = game.reset();
    for (int step = 0; step < 100; step++) {
        rows[s.y][s.x] = '*';
        auto move = games::greedy<game_type>(memory, s, game.actions(s), prng);
        auto out  = game.step(move);
        s = out.next;
        if (out.terminated) {
            break;
        }
    }
    // mark the start and the goal, then print the grid and the wind below it
    rows[game_type::start().y][game_type::start().x] = 'S';
    rows[game_type::goal().y][game_type::goal().x]   = 'G';
    for (const auto & row : rows) {
        std::cout << "    " << row << std::endl;
    }
    std::cout << "    ";
    for (int x = 0; x < game_type::width; x++) {
        std::cout << game_type::wind(x);
    }
    std::cout << "   <- wind" << std::endl;
}

int main()
{
    std::mt19937 prng(1);

    // 1. steady wind: Q-learning should find the 15-move shortest path
    game_type steady;
    relearn::q_learning<types::state, types::action> q_steady;
    q_steady.alpha = 0.5;
    q_steady.gamma = 1.0;
    auto memory = train(steady, q_steady, 500, prng);
    // every move costs -1, so the number of moves is minus the return
    double moves = -games::evaluate(steady, memory, 1, 100, prng);
    std::cout << std::fixed << std::setprecision(0)
              << "steady wind, q_learning: " << moves << " moves (the shortest path is 15)" << std::endl;
    draw(steady, memory, prng);

    // 2. random wind: compare Q-learning, with a constant and with a decaying
    //    learning rate, and probabilistic Q-learning, each trained for 5000 episodes
    game_type random_wind(true, 2);
    relearn::q_learning<types::state, types::action> q_constant;
    q_constant.alpha = 0.5;
    q_constant.gamma = 1.0;
    relearn::q_learning<types::state, types::action> q_decaying;
    q_decaying.gamma = 1.0;
    // α = 1/n^0.6 for the n-th update of each state/action pair
    q_decaying.learning_rate = relearn::polynomial_decay<>(1.0, 0.6);
    relearn::q_probabilistic<types::state, types::action> p_random(1.0);
    auto memory_constant = train(random_wind, q_constant, 5000, prng);
    auto memory_decaying = train(random_wind, q_decaying, 5000, prng);
    auto memory_p        = train(random_wind, p_random, 5000, prng);
    // a policy that wanders forever is stopped at 1000 moves
    double moves_constant = -games::evaluate(random_wind, memory_constant, 1000, 1000, prng);
    double moves_decaying = -games::evaluate(random_wind, memory_decaying, 1000, 1000, prng);
    double moves_p        = -games::evaluate(random_wind, memory_p, 1000, 1000, prng);
    std::cout << std::endl << "random wind, average moves over 1000 greedy episodes"
              << " (the best policy needs 20.6 on average):" << std::endl;
    std::cout << std::setprecision(1);
    std::cout << "    q_learning, α = 0.5           " << std::setw(7) << moves_constant << std::endl;
    std::cout << "    q_learning, α = 1/n^0.6       " << std::setw(7) << moves_decaying << std::endl;
    std::cout << "    q_probabilistic               " << std::setw(7) << moves_p << std::endl;
    // the steady path must be the shortest, and the decaying learning rate must
    // come close to the best policy in random wind
    bool learnt = (moves == 15) && (moves_decaying < 21.6);
    return learnt ? EXIT_SUCCESS : EXIT_FAILURE;
}
