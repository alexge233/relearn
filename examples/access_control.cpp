/**
 * The access-control queuing task (Sutton & Barto, 1st edition Example 6.7, 2nd edition Example 10.2),
 * a task that never ends, learnt after every step, in three ways:
 *
 * 1. R-Learning with the 1st edition's settings, α = 0.1 and β = 0.01 (Figure 6.17)
 * 2. R-Learning with a smaller learning rate, α = 0.01
 * 3. differential Sarsa with the 2nd edition's settings, α = 0.01 and β = 0.01 (Figure 10.5)
 *
 * all exploring ε-greedy with ε = 0.1, for 2000000 steps. For each one it prints the average reward ρ
 * it learnt, the greedy policy and the values learnt, in the layout of the book's figures, and what the
 * greedy policy earns per step: worked out exactly from the rules, and by playing it for 1000000 steps.
 *
 * The optimal policy, and its average reward ρ* = 2.7476, are worked out exactly by relative value
 * iteration in tests/solvers.hpp (and checked in tests/r_learning_test.cpp).
 *
 * The program succeeds if R-Learning's ρ (with the book's settings) is within 0.05 of ρ*, and above
 * what the agent earns whilst exploring, and if the greedy policies of R-Learning with α = 0.01 and of
 * differential Sarsa earn within 0.08 of ρ*.
 *
 * @see games/access_control.hpp for the rules
 */
#include "games/access_control.hpp"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// set shortcuts to the game and to its relearn types
using game_type = games::access_control;
using types     = games::types<game_type>;

/// what one learner learnt on the queue
struct learnt
{
    /// the values learnt
    types::policy memory;
    /// the learner's ρ at the end of the run
    double rho = 0;
    /// the mean of ρ over the second half of the run (ρ moves a lot from one step to the next)
    double mean_rho = 0;
    /// the mean reward per step earned over the second half of the run, whilst exploring
    double reward = 0;
    /// the number of visits of each state: [free servers][priority 1, 2, 4, 8]
    std::vector<std::vector<std::size_t>> visits = std::vector<std::vector<std::size_t>>(11, std::vector<std::size_t>(4, 0));
};

/// @return the column of @param priority: 0 for 1, 1 for 2, 2 for 4 and 3 for 8 (or anything else)
unsigned int column(unsigned int priority)
{
    unsigned int k = 0;
    while (k < 3 && (1u << k) != priority) {
        k++;
    }
    return k;
}

/**
 * @brief learn the queue, after every step
 * @param learner is `r_learning` or `differential_sarsa`
 * @param queue is the queue to learn
 * @param steps is the number of steps
 * @param prng chooses the exploring actions, and breaks ties
 * @return what was learnt
 *
 * The queue never ends, so it is played in pieces of 1000 steps: its `reset()` does not
 * start again, so each piece carries on where the last one stopped.
 */
template <class learner_type>
learnt learn(learner_type & learner,
             game_type & queue,
             std::size_t steps,
             std::mt19937 & prng)
{
    learnt result;
    // the ε-greedy agent, with ε = 0.1
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(result.memory, s, legal, 0.1, prng);
    };
    std::size_t pieces = 0;
    for (std::size_t done = 0; done < steps; done += 1000) {
        // 1000 steps, learning after each one
        auto played = games::play_online(queue, agent, learner, result.memory, 1000);
        // count the states visited (the last link is the state reached, not a step)
        for (std::size_t i = 0; i + 1 < played.episode.size(); i++) {
            auto s = played.episode[i].state.trait();
            result.visits[s.free][column(s.priority)]++;
        }
        // over the second half, average ρ and the reward earned
        if (done >= steps / 2) {
            result.mean_rho += learner.rho;
            result.reward += played.total_reward / 1000;
            pieces++;
        }
    }
    result.rho = learner.rho;
    result.mean_rho /= static_cast<double>(pieces);
    result.reward /= static_cast<double>(pieces);
    return result;
}

/// @return the greedy action learnt in @param s: accept if its value is higher, otherwise reject
unsigned int greedy_action(const types::policy & memory,
                           const game_type::state_type & s)
{
    if (s.free == 0) {
        return game_type::reject;
    }
    double q_accept = memory.value(types::state(s), types::action(game_type::accept));
    double q_reject = memory.value(types::state(s), types::action(game_type::reject));
    return q_accept > q_reject ? game_type::accept : game_type::reject;
}

/// @return the optimal action in @param s, as worked out in tests/solvers.hpp: always accept 8 and 4,
///         accept 2 with at least 4 free servers, never accept 1
unsigned int optimal_action(const game_type::state_type & s)
{
    const bool accept = s.free > 0 && (s.priority >= 4 || (s.priority == 2 && s.free >= 4));
    return accept ? game_type::accept : game_type::reject;
}

/**
 * @brief print a policy as in Figures 6.17 and 10.5: one row per priority, one column per free servers
 * @param choose gives the action of each state
 * @param visits marks in lower case the states visited fewer than 1000 times (empty: no marks)
 */
template <class chooser>
void print_policy(chooser && choose,
                  const std::vector<std::vector<std::size_t>> & visits)
{
    std::cout << "    priority \\ free servers:  1  2  3  4  5  6  7  8  9 10" << std::endl;
    for (unsigned int priority : {8u, 4u, 2u, 1u}) {
        std::cout << "    " << std::setw(23) << priority << ":";
        for (unsigned int free = 1; free <= 10; free++) {
            bool accept = choose(game_type::state_type{free, priority}) == game_type::accept;
            bool rare = !visits.empty() && visits[free][column(priority)] < 1000;
            char mark = accept ? 'A' : 'R';
            std::cout << "  " << (rare ? static_cast<char>(mark - 'A' + 'a') : mark);
        }
        std::cout << std::endl;
    }
}

/**
 * @brief print the value of the best action of each state, as in the lower part of Figures 6.17 and 10.5
 * @param memory is the learnt policy
 */
void print_values(const types::policy & memory)
{
    std::cout << "    value of the best action, free servers:" << std::endl;
    std::cout << "    priority:";
    for (unsigned int free = 0; free <= 10; free++) {
        std::cout << std::setw(7) << free;
    }
    std::cout << std::endl;
    for (unsigned int priority : {8u, 4u, 2u, 1u}) {
        std::cout << "    " << std::setw(8) << priority << ":";
        for (unsigned int free = 0; free <= 10; free++) {
            game_type::state_type s{free, priority};
            double best = memory.value(types::state(s), types::action(greedy_action(memory, s)));
            std::cout << std::setw(7) << std::setprecision(1) << best << std::setprecision(4);
        }
        std::cout << std::endl;
    }
}

/**
 * @brief print what a learner learnt
 * @param title names the learner
 * @param result is what it learnt
 * @param queue is the queue it learnt, to play the greedy policy on
 * @return the exact average reward per step of the greedy policy learnt
 */
double report(const std::string & title,
              const learnt & result,
              game_type & queue)
{
    std::cout << std::endl << title << std::endl;
    std::cout << "    ρ learnt: " << result.rho
              << ", and its mean over the second half: " << result.mean_rho << std::endl;
    std::cout << "    earned per step whilst exploring (second half): " << result.reward << std::endl;
    std::cout << "    greedy policy learnt (lower case: states seen fewer than 1000 times):" << std::endl;
    auto greedy = [&](const game_type::state_type & s) {
        return greedy_action(result.memory, s);
    };
    print_policy(greedy, result.visits);
    print_values(result.memory);
    // what the greedy policy earns: exactly, and by playing it for a while
    double exact = queue.average_reward(greedy);
    double played = 0;
    for (unsigned int i = 0; i < 1000000; i++) {
        played += queue.step(greedy(queue.reset())).reward;
    }
    std::cout << "    the greedy policy earns per step, exactly: " << exact
              << ", playing it for 1000000 steps: " << played / 1000000 << std::endl;
    return exact;
}

int main()
{
    const std::size_t steps = 2000000;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "The access-control queue: 10 servers, each busy one free again with probability 0.06 per step,"
              << std::endl << "learnt for " << steps << " steps, ε-greedy with ε = 0.1" << std::endl;

    // the optimal policy, from tests/solvers.hpp, and what it earns, worked out from the rules
    game_type rules;
    const double optimal = rules.average_reward(optimal_action);
    std::cout << std::endl << "the optimal policy (A = accept, R = reject) earns " << optimal << " per step:" << std::endl;
    print_policy(optimal_action, {});
    std::cout << "    accepting every customer earns " << rules.average_reward([](const game_type::state_type &) {
                     return game_type::accept;
                 }) << std::endl;

    // 1. R-Learning, with the 1st edition's settings
    game_type queue_1(10, 0.06, 1);
    std::mt19937 prng_1(101);
    relearn::r_learning<types::state, types::action> r_learning(0.1, 0.01);
    // its maxima count the actions not tried yet as 0, like the agent does
    auto legal_actions = [&](const types::state & s) {
        std::vector<types::action> legal;
        for (const auto & a : queue_1.actions(s.trait())) {
            legal.push_back(types::action(a));
        }
        return legal;
    };
    r_learning.legal_actions = legal_actions;
    auto book = learn(r_learning, queue_1, steps, prng_1);
    report("R-Learning, α = 0.1, β = 0.01 (1st edition, Example 6.7, Figure 6.17):", book, queue_1);

    // 2. R-Learning, with a smaller learning rate
    game_type queue_2(10, 0.06, 2);
    std::mt19937 prng_2(102);
    relearn::r_learning<types::state, types::action> r_small(0.01, 0.01);
    r_small.legal_actions = legal_actions;
    auto small = learn(r_small, queue_2, steps, prng_2);
    double small_exact = report("R-Learning, α = 0.01, β = 0.01:", small, queue_2);

    // 3. differential Sarsa, with the 2nd edition's settings
    game_type queue_3(10, 0.06, 3);
    std::mt19937 prng_3(103);
    relearn::differential_sarsa<types::state, types::action> sarsa(0.01, 0.01);
    auto second = learn(sarsa, queue_3, steps, prng_3);
    double sarsa_exact = report("differential Sarsa, α = 0.01, β = 0.01 (2nd edition, Example 10.2, Figure 10.5):",
                                second, queue_3);

    // R-Learning's ρ is the greedy policy's, not the exploring one's; both greedy policies are close to optimal
    const bool good = std::fabs(book.mean_rho - optimal) < 0.05 &&
                      book.mean_rho > book.reward + 0.1 &&
                      optimal - small_exact < 0.08 &&
                      optimal - sarsa_exact < 0.08;
    return good ? EXIT_SUCCESS : EXIT_FAILURE;
}
