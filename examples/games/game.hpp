#ifndef RELEARN_GAMES_GAME_HPP
#define RELEARN_GAMES_GAME_HPP
/**
 * Common pieces shared by the example games.
 *
 * Every game is a class with the same shape (a convention, not a base class):
 *
 *     using state_type  = ...;   // hashable (std::hash) and comparable (==)
 *     using action_type = ...;   // hashable and comparable
 *     state_type reset();                                    // start a new episode
 *     step_result<state_type> step(const action_type & a);   // take one action
 *     std::vector<action_type> actions(const state_type & s) const; // legal actions
 *
 * This mirrors Gymnasium's reset()/step(), so the games can later move behind
 * a relearn environment interface without changing their rules.
 *
 * Episodes are recorded in relearn's reward convention: the link (s_t, a_t)
 * carries in `s_t.reward()` the reward received after taking a_t in s_t.
 */
#include "../../src/relearn.hpp"
#include <cstddef>
#include <deque>
#include <random>
#include <vector>

namespace games
{
/// outcome of one step of a game
template <class state_type>
struct step_result
{
    state_type next;            // state after the step
    double reward = 0;          // reward received for the step
    bool terminated = false;    // the game ended (goal reached, game lost, ...)
};

/// the relearn types used to learn a game
template <class game_type>
struct types
{
    using state   = relearn::state<typename game_type::state_type>;
    using action  = relearn::action<typename game_type::action_type>;
    using link    = relearn::link<state, action>;
    using episode = std::deque<link>;
    using policy  = relearn::policy<state, action>;
};

/// one played episode
template <class game_type>
struct played
{
    /// the links, ready for a learner. An episode cut short ends with one extra
    /// link holding the state reached: it is not a step taken, and only gives
    /// the learner the next state of the last real step
    typename types<game_type>::episode episode;
    double total_reward = 0;                     // undiscounted sum of rewards
    std::size_t steps = 0;                       // actions taken
    bool terminated = false;                     // false: cut short by the step limit
};

/**
 * @brief play one episode of @param game
 * @param choose is called as `choose(state, legal_actions)` and returns an action
 * @param max_steps cuts the episode short. Pass `terminated` on to the learner,
 *        `learner(result.episode, policy, result.terminated)`, so that a cut-off
 *        is not learnt as the end of the game; the last step taken is still
 *        learnt, from the value of the state it reached
 */
template <class game_type, class chooser>
played<game_type> play(game_type & game, chooser && choose, std::size_t max_steps)
{
    using state  = typename types<game_type>::state;
    using action = typename types<game_type>::action;
    played<game_type> result;
    auto s_t = game.reset();
    while (result.steps < max_steps) {
        auto a_t = choose(s_t, game.actions(s_t));
        auto out = game.step(a_t);
        result.episode.push_back({state(out.reward, s_t), action(a_t)});
        result.total_reward += out.reward;
        result.steps++;
        s_t = out.next;
        if (out.terminated) {
            result.terminated = true;
            break;
        }
    }
    if (!result.terminated && result.steps > 0) {
        // cut short: add the state reached, so that the last step taken can be
        // learnt from it (a cut-short episode's last link is never learnt itself)
        auto legal = game.actions(s_t);
        if (!legal.empty()) {
            result.episode.push_back({state(0, s_t), action(legal.front())});
        }
    }
    return result;
}

/**
 * @return the legal action with the highest Q-value in @param s_t.
 * Actions never tried count as 0 (they are not stored in the policy yet),
 * and ties are broken at random so that equal values are all explored.
 */
template <class game_type>
typename game_type::action_type greedy(const typename types<game_type>::policy & memory,
                                       const typename game_type::state_type & s_t,
                                       const std::vector<typename game_type::action_type> & legal,
                                       std::mt19937 & prng)
{
    using state  = typename types<game_type>::state;
    using action = typename types<game_type>::action;
    std::vector<typename game_type::action_type> best;
    double best_value = 0;
    for (const auto & a : legal) {
        double q = memory.value(state(s_t), action(a));
        if (best.empty() || q > best_value) {
            best.assign(1, a);
            best_value = q;
        }
        else if (q == best_value) {
            best.push_back(a);
        }
    }
    std::uniform_int_distribution<std::size_t> pick(0, best.size() - 1);
    return best[pick(prng)];
}

/// with probability @param epsilon a random legal action, otherwise the greedy one
template <class game_type>
typename game_type::action_type epsilon_greedy(const typename types<game_type>::policy & memory,
                                               const typename game_type::state_type & s_t,
                                               const std::vector<typename game_type::action_type> & legal,
                                               double epsilon,
                                               std::mt19937 & prng)
{
    std::bernoulli_distribution explore(epsilon);
    if (explore(prng)) {
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        return legal[pick(prng)];
    }
    return greedy<game_type>(memory, s_t, legal, prng);
}

/// @return the average undiscounted return of the greedy policy over @param episodes games
template <class game_type>
double evaluate(game_type & game,
                const typename types<game_type>::policy & memory,
                std::size_t episodes,
                std::size_t max_steps,
                std::mt19937 & prng)
{
    double sum = 0;
    for (std::size_t i = 0; i < episodes; i++) {
        sum += play(game,
                    [&](const typename game_type::state_type & s,
                        const std::vector<typename game_type::action_type> & legal) {
                        return greedy<game_type>(memory, s, legal, prng);
                    },
                    max_steps).total_reward;
    }
    return sum / static_cast<double>(episodes);
}
} // namespace games
#endif
