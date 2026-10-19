/**
 * Cliff Walking (Sutton & Barto, Example 6.6): Q-learning, SARSA and Expected SARSA.
 *
 * The three learners play the same game, exploring with ε-greedy (ε = 0.1),
 * and learn after every step, as in the book (α = 0.5, γ = 1).
 *
 * - Q-learning learns the values of the *best* policy: the shortest path, right
 *   along the edge of the cliff. But the agent still explores while it plays,
 *   and a random move along the edge throws it off the cliff now and then.
 * - SARSA learns the values of the policy it *follows*, exploration included,
 *   so it learns that the edge is dangerous, and takes a safer path further away.
 *   While exploring, it earns more than Q-learning.
 * - Expected SARSA averages over the moves the policy may take instead of using
 *   the one it took, so it learns more steadily, and earns the most.
 *
 * Each learner is trained in 20 runs, and the returns are averaged over the runs,
 * as the book does.
 *
 * @see games/cliff_walking.hpp for the rules
 * @see docs/sarsa.md for the algorithms
 */
#include "games/cliff_walking.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

// set shortcuts to the game and to its relearn types
using game_type = games::cliff_walking;
using types     = games::types<game_type>;

// the settings of Sutton & Barto, Example 6.6
const double epsilon = 0.1;
const double alpha   = 0.5;
const unsigned int runs     = 20;
const unsigned int episodes = 500;

/**
 * @brief prepare a learner: nothing to do for Q-learning and SARSA
 * @param learner is the learner
 * @param memory is the policy being learnt
 */
template <class learner_type>
void prepare(learner_type &,
             const types::policy &)
{}

/**
 * @brief prepare Expected SARSA: give it the ε-greedy policy π the agent follows
 * @param learner is the learner
 * @param memory is the policy being learnt, which the agent is greedy on
 */
void prepare(relearn::expected_sarsa<types::state,types::action> & learner,
             const types::policy & memory)
{
    game_type rules;
    learner.probabilities = [&memory, rules](const types::state & s) {
        return games::epsilon_greedy_probabilities<game_type>(memory,
                                                              s.trait(),
                                                              rules.actions(s.trait()),
                                                              epsilon);
    };
}

/**
 * @brief draw the grid, marking with `*` the cells the greedy policy visits
 * @param memory is the learnt policy
 * @param prng breaks ties between actions of equal value
 */
void draw(const types::policy & memory,
          std::mt19937 & prng)
{
    // an empty grid, with the cliff along the bottom row
    std::vector<std::string> rows(game_type::height, std::string(game_type::width, '.'));
    for (int x = 1; x < game_type::width - 1; x++) {
        rows[game_type::height - 1][x] = 'C';
    }
    // follow the greedy policy from the start, marking each cell visited
    game_type game;
    auto s = game.reset();
    bool reached = false;
    for (int step = 0; step < 100; step++) {
        rows[s.y][s.x] = '*';
        auto move = games::greedy<game_type>(memory, s, game.actions(s), prng);
        auto out  = game.step(move);
        if (out.terminated) {
            reached = true;
            break;
        }
        // a greedy policy that comes back to a cell it has visited goes round in circles
        if (rows[out.next.y][out.next.x] == '*') {
            break;
        }
        s = out.next;
    }
    // mark the start and the goal, then print the grid
    rows[game_type::start().y][game_type::start().x] = 'S';
    rows[game_type::goal().y][game_type::goal().x]   = 'G';
    for (const auto & row : rows) {
        std::cout << "    " << row << std::endl;
    }
    if (!reached) {
        std::cout << "    (the greedy policy goes round in circles: with a constant α"
                  << " its values are still noisy)" << std::endl;
    }
}

/**
 * @brief train a learner in `runs` runs of `episodes` episodes each
 * @param name is the learner's name, to print
 * @param returns gets, for each episode, its return while exploring, averaged over the runs
 * @return the average return while exploring over episodes 101 ... 500
 */
template <class learner_type>
double train(const std::string & name,
             std::vector<double> & returns)
{
    returns.assign(episodes, 0.0);
    for (unsigned int run = 1; run <= runs; run++) {
        // a new agent for each run: an empty policy and its own random numbers
        game_type game;
        std::mt19937 prng(run);
        types::policy memory;
        learner_type learner;
        learner.alpha = alpha;
        learner.gamma = 1.0;
        prepare(learner, memory);

        // the ε-greedy agent: explores with probability ε, otherwise takes the best known move
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, epsilon, prng);
        };
        // play and learn after every step, adding up each episode's return
        for (unsigned int i = 0; i < episodes; i++) {
            auto r = games::play_online(game, agent, learner, memory, 1000);
            returns[i] += r.total_reward / runs;
        }
        // draw the path the first run has learnt
        if (run == 1) {
            std::cout << name << ", the greedy path learnt in the first run:" << std::endl;
            draw(memory, prng);
            std::cout << std::endl;
        }
    }
    // the average return once the learning has settled, after episode 100
    double sum = 0;
    for (unsigned int i = 100; i < episodes; i++) {
        sum += returns[i];
    }
    return sum / (episodes - 100);
}

int main()
{
    // train each learner, keeping the average return of each episode
    std::vector<double> q;
    std::vector<double> sarsa;
    std::vector<double> expected;
    double q_mean        = train<relearn::q_learning<types::state,types::action>>("Q-learning", q);
    double sarsa_mean    = train<relearn::sarsa<types::state,types::action>>("SARSA", sarsa);
    double expected_mean = train<relearn::expected_sarsa<types::state,types::action>>("Expected SARSA", expected);

    // print the average return while exploring, in blocks of 50 episodes
    std::cout << "average return while exploring (" << runs << " runs):" << std::endl;
    std::cout << "    episodes   Q-learning     SARSA   Expected SARSA" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    for (unsigned int block = 0; block < episodes / 50; block++) {
        double sums[3] = {0, 0, 0};
        for (unsigned int i = block * 50; i < (block + 1) * 50; i++) {
            sums[0] += q[i] / 50;
            sums[1] += sarsa[i] / 50;
            sums[2] += expected[i] / 50;
        }
        std::cout << "     " << std::setw(3) << block * 50 + 1 << "-" << std::setw(3) << (block + 1) * 50
                  << "   " << std::setw(10) << sums[0]
                  << "   " << std::setw(7) << sums[1]
                  << "   " << std::setw(14) << sums[2] << std::endl;
    }
    std::cout << "     101-500   " << std::setw(10) << q_mean
              << "   " << std::setw(7) << sarsa_mean
              << "   " << std::setw(14) << expected_mean << std::endl;

    // as in the book: SARSA earns more than Q-learning, and Expected SARSA the most
    bool as_expected = (sarsa_mean > q_mean) && (expected_mean > sarsa_mean);
    return as_expected ? EXIT_SUCCESS : EXIT_FAILURE;
}
