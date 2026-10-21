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
#include <algorithm>
#include <cstddef>
#include <deque>
#include <random>
#include <vector>

namespace games
{
/**
 * @struct step_result
 * @brief the outcome of one step of a game: the state reached, the reward, and whether the game ended
 */
template <class state_type>
struct step_result
{
    /// the state after the step
    state_type next;
    /// the reward received for the step
    double reward = 0;
    /// true if the game ended (goal reached, game lost, ...)
    bool terminated = false;
};

/**
 * @struct types
 * @brief the relearn types used to learn a game: its states, actions, links, episodes and policy
 */
template <class game_type>
struct types
{
    using state   = relearn::state<typename game_type::state_type>;
    using action  = relearn::action<typename game_type::action_type>;
    using link    = relearn::link<state, action>;
    using episode = std::deque<link>;
    using policy  = relearn::policy<state, action>;
};

/**
 * @struct played
 * @brief one played episode, ready for a learner
 */
template <class game_type>
struct played
{
    /// the links, ready for a learner. An episode cut short ends with one extra
    /// link holding the state reached: it is not a step taken, and only gives
    /// the learner the next state of the last real step
    typename types<game_type>::episode episode;
    /// the undiscounted sum of rewards
    double total_reward = 0;
    /// the number of actions taken
    std::size_t steps = 0;
    /// false if the episode was cut short by the step limit
    bool terminated = false;
};

/**
 * @brief play one episode of @param game
 * @param choose is called as `choose(state, legal_actions)` and returns an action
 * @param max_steps cuts the episode short. Pass `terminated` on to the learner,
 *        `learner(result.episode, policy, result.terminated)`, so that a cut-off
 *        is not learnt as the end of the game; the last step taken is still
 *        learnt, from the value of the state it reached
 * @return the played episode
 */
template <class game_type,
          class chooser>
played<game_type> play(game_type & game,
                       chooser && choose,
                       std::size_t max_steps)
{
    using state  = typename types<game_type>::state;
    using action = typename types<game_type>::action;
    played<game_type> result;
    // start a new game
    auto s_t = game.reset();
    while (result.steps < max_steps) {
        // choose an action and take it
        auto a_t = choose(s_t, game.actions(s_t));
        auto out = game.step(a_t);
        // record (s_t, a_t) with the reward received after taking a_t
        result.episode.push_back({state(out.reward, s_t), action(a_t)});
        result.total_reward += out.reward;
        result.steps++;
        // move on to the state reached
        s_t = out.next;
        if (out.terminated) {
            result.terminated = true;
            break;
        }
    }
    if (!result.terminated && result.steps > 0) {
        // cut short: add the state reached, so that the last step taken can be
        // learnt from it (a cut-short episode's last link is never learnt itself).
        // Its action is the one the agent would take next, which SARSA learns from
        auto legal = game.actions(s_t);
        if (!legal.empty()) {
            auto a_next = choose(s_t, legal);
            result.episode.push_back({state(0, s_t), action(a_next)});
        }
    }
    return result;
}

/**
 * @brief the last steps of an episode, for `play_online`
 * @param steps is the episode played so far
 * @param count is the number of steps wanted (at least 1)
 * @return the last @param count links of @param steps, or all of them if there are fewer
 */
template <class episode_type>
episode_type last_steps(const episode_type & steps,
                        std::size_t count)
{
    const std::size_t wanted = std::max<std::size_t>(count, 1);
    const std::size_t first  = steps.size() > wanted ? steps.size() - wanted : 0;
    episode_type recent;
    for (std::size_t i = first; i < steps.size(); i++) {
        recent.push_back(steps[i]);
    }
    return recent;
}

/**
 * @brief a learner that keeps something from one call to the next (as `relearn::true_online_sarsa_lambda`
 *        its traces) is told that a new episode starts: `learner.start_episode()`
 */
template <class learner_type>
auto start_episode(learner_type & learner, int) -> decltype(learner.start_episode(), void())
{
    learner.start_episode();
}

/// @brief the other learners start every call afresh, and have nothing to be told
template <class learner_type>
void start_episode(learner_type &, long)
{}

/**
 * @brief play one episode of @param game, learning after every step (online)
 * @param choose is called as `choose(state, legal_actions)` and returns an action
 * @param learner is any relearn learner: `q_learning`, `sarsa`, `expected_sarsa`, the n-step learners,
 *        `true_online_sarsa_lambda`
 * @param memory is the policy being learnt (and used by @param choose)
 * @param max_steps cuts the episode short
 * @param window is the number of steps the learner is given after each step:
 *        1 (the default) for the one-step learners, n for `n_step_sarsa` and `n_step_td`
 * @return the played episode, already learnt
 *
 * Sutton & Barto's TD algorithms learn after every step, while the episode is
 * played, so that what the agent learns changes how it plays at once.
 * After each step, the learner is given two links: (s_t, a_t) with its reward,
 * and (s_{t+1}, a_{t+1}) with the action chosen next, and `terminated = false`.
 * It learns step t from the next state and action, and not yet step t+1.
 * The next action is chosen before learning, as in the book's SARSA.
 * When the game ends, the last step is learnt alone, as the end of the game.
 * A learner with a `start_episode()` (as `relearn::true_online_sarsa_lambda`) is told before
 * the game starts, even after a game cut short.
 *
 * With a @param window of n, the learner is given the last n steps (fewer at the
 * start) and then (s_{t+1}, a_{t+1}), so that an n-step learner learns step t-n+1;
 * when the game ends, it is given the last n steps, which it learns as the end of the game.
 *
 * @warning the window must match the learner: 1 for the one-step learners, `learner.n` for the n-step
 *          ones. Nothing checks it. With a smaller window an n-step learner learns no step until the end
 *          of the game, and then only the last ones; with a larger one, a learner learns the same steps
 *          again after every step.
 */
template <class game_type,
          class chooser,
          class learner_type>
played<game_type> play_online(game_type & game,
                              chooser && choose,
                              learner_type & learner,
                              typename types<game_type>::policy & memory,
                              std::size_t max_steps,
                              std::size_t window = 1)
{
    using state   = typename types<game_type>::state;
    using action  = typename types<game_type>::action;
    using episode = typename types<game_type>::episode;
    played<game_type> result;
    // start a new game, and choose the first action; a learner that keeps traces starts them again
    start_episode(learner, 0);
    auto s_t = game.reset();
    auto a_t = choose(s_t, game.actions(s_t));
    while (result.steps < max_steps) {
        // take the action, and record (s_t, a_t) with the reward received
        auto out = game.step(a_t);
        result.episode.push_back({state(out.reward, s_t), action(a_t)});
        result.total_reward += out.reward;
        result.steps++;
        if (out.terminated) {
            // the game ended: learn the last step (or the last `window` steps), as the end of the game
            result.terminated = true;
            episode last = last_steps(result.episode, window);
            learner(last, memory, true);
            break;
        }
        // choose the next action first: SARSA learns step t from it
        auto a_next = choose(out.next, game.actions(out.next));
        // learn step t from (s_{t+1}, a_{t+1}), without learning step t+1 yet
        // (an n-step learner, given the last n steps, learns step t-n+1)
        episode recent = last_steps(result.episode, window);
        recent.push_back({state(0, out.next), action(a_next)});
        learner(recent, memory, false);
        // move on to the next state and action
        s_t = out.next;
        a_t = a_next;
    }
    if (!result.terminated && result.steps > 0) {
        // cut short: as in `play`, end with the state reached and the action chosen next
        result.episode.push_back({state(0, s_t), action(a_t)});
    }
    return result;
}

/**
 * @brief play one episode of @param game, learning after every step, with one-step Actor-Critic
 * @param choose is called as `choose(state, legal_actions)` and returns an action
 * @param learner is `relearn::actor_critic`
 * @param memory are the preferences being learnt (and used by @param choose)
 * @param max_steps cuts the episode short
 * @return the played episode, already learnt
 *
 * The same as `play_online`, except for when each action is chosen. SARSA learns step t from
 * the action taken next, so `play_online` chooses it before learning; Actor-Critic draws each
 * action from the policy just updated (Sutton & Barto, Section 13.5), so here every action is
 * chosen after the step before has been learnt.
 * After each step, the learner is given two links: (s_t, a_t) with its reward, and the state
 * reached, s_{t+1}, with `terminated = false`. The action of that second link is a_t again, a
 * placeholder which Actor-Critic does not use. When the game ends, the last step is learnt alone.
 * The episode starts with `learner.start_episode()`, so that I = 1 even after an episode cut short.
 */
template <class game_type,
          class chooser,
          class learner_type>
played<game_type> play_online_actor_critic(game_type & game,
                                           chooser && choose,
                                           learner_type & learner,
                                           typename types<game_type>::policy & memory,
                                           std::size_t max_steps)
{
    using state   = typename types<game_type>::state;
    using action  = typename types<game_type>::action;
    using episode = typename types<game_type>::episode;
    played<game_type> result;
    // start a new game, and a new episode for the learner: I = 1
    learner.start_episode();
    auto s_t = game.reset();
    while (result.steps < max_steps) {
        // choose an action with the policy learnt so far, take it,
        // and record (s_t, a_t) with the reward received
        auto a_t = choose(s_t, game.actions(s_t));
        auto out = game.step(a_t);
        result.episode.push_back({state(out.reward, s_t), action(a_t)});
        result.total_reward += out.reward;
        result.steps++;
        if (out.terminated) {
            // the game ended: learn the last step alone, as the end of the game
            result.terminated = true;
            episode last = {result.episode.back()};
            learner(last, memory, true);
            break;
        }
        // learn step t from the state reached (its action, a_t, is only a placeholder)
        episode window = {result.episode.back(), {state(0, out.next), action(a_t)}};
        learner(window, memory, false);
        // move on to the state reached
        s_t = out.next;
    }
    if (!result.terminated && result.steps > 0) {
        // cut short: as in `play`, end with the state reached and the action the agent would take next
        auto legal = game.actions(s_t);
        if (!legal.empty()) {
            result.episode.push_back({state(0, s_t), action(choose(s_t, legal))});
        }
    }
    return result;
}

/**
 * @brief the greedy choice: the legal action with the highest Q-value
 * @param memory is the learnt policy
 * @param s_t is the current state
 * @param legal are the legal actions in @param s_t
 * @param prng breaks ties between actions of equal value
 * @return the legal action with the highest Q-value in @param s_t.
 * @note actions never tried count as 0 (they are not stored in the policy yet),
 *       and ties are broken at random so that equal values are all explored.
 */
template <class game_type>
typename game_type::action_type greedy(const typename types<game_type>::policy & memory,
                                       const typename game_type::state_type & s_t,
                                       const std::vector<typename game_type::action_type> & legal,
                                       std::mt19937 & prng)
{
    using state  = typename types<game_type>::state;
    using action = typename types<game_type>::action;
    // the actions sharing the highest value found so far
    std::vector<typename game_type::action_type> best;
    double best_value = 0;
    for (const auto & a : legal) {
        double q = memory.value(state(s_t), action(a));
        if (best.empty() || q > best_value) {
            // a new highest value: it is the only best action so far
            best.assign(1, a);
            best_value = q;
        }
        else if (q == best_value) {
            // a tie with the highest value
            best.push_back(a);
        }
    }
    // pick one of the best actions at random
    std::uniform_int_distribution<std::size_t> pick(0, best.size() - 1);
    return best[pick(prng)];
}

/**
 * @brief the ε-greedy choice: explore with probability @param epsilon, otherwise be greedy
 * @param memory is the learnt policy
 * @param s_t is the current state
 * @param legal are the legal actions in @param s_t
 * @param prng draws the random choices
 * @return a random legal action with probability @param epsilon, otherwise the greedy one
 */
template <class game_type>
typename game_type::action_type epsilon_greedy(const typename types<game_type>::policy & memory,
                                               const typename game_type::state_type & s_t,
                                               const std::vector<typename game_type::action_type> & legal,
                                               double epsilon,
                                               std::mt19937 & prng)
{
    std::bernoulli_distribution explore(epsilon);
    if (explore(prng)) {
        // explore: any legal action, chosen at random
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        return legal[pick(prng)];
    }
    // exploit: the best known action
    return greedy<game_type>(memory, s_t, legal, prng);
}

/**
 * @brief the probabilities of the ε-greedy choice, π(a | s), as Expected SARSA needs them
 * @param memory is the learnt policy
 * @param s_t is the current state
 * @param legal are the legal actions in @param s_t
 * @param epsilon is the probability of exploring
 * @return each legal action with the probability that `epsilon_greedy` takes it:
 *         ε / |A| for every action, plus (1 - ε) shared between the best actions
 * @note like `greedy`, actions never tried count as 0, and tied best actions
 *       share the greedy probability equally
 */
template <class game_type>
std::vector<std::pair<typename types<game_type>::action,double>>
    epsilon_greedy_probabilities(const typename types<game_type>::policy & memory,
                                 const typename game_type::state_type & s_t,
                                 const std::vector<typename game_type::action_type> & legal,
                                 double epsilon)
{
    using state  = typename types<game_type>::state;
    using action = typename types<game_type>::action;
    // find the highest value, and how many actions share it
    double best_value = 0;
    std::size_t ties = 0;
    for (const auto & a : legal) {
        double q = memory.value(state(s_t), action(a));
        if (ties == 0 || q > best_value) {
            best_value = q;
            ties = 1;
        }
        else if (q == best_value) {
            ties++;
        }
    }
    // every action is explored with probability ε / |A|,
    // and the best actions share the remaining 1 - ε
    std::vector<std::pair<action,double>> probabilities;
    const double explore = epsilon / static_cast<double>(legal.size());
    for (const auto & a : legal) {
        double p = explore;
        if (memory.value(state(s_t), action(a)) == best_value) {
            p += (1.0 - epsilon) / static_cast<double>(ties);
        }
        probabilities.push_back(std::make_pair(action(a), p));
    }
    return probabilities;
}

/**
 * @brief measure the greedy policy
 * @param game is the game to play
 * @param memory is the learnt policy
 * @param episodes is the number of games to play
 * @param max_steps cuts each game short
 * @param prng breaks ties between actions of equal value
 * @return the average undiscounted return of the greedy policy over @param episodes games
 */
template <class game_type>
double evaluate(game_type & game,
                const typename types<game_type>::policy & memory,
                std::size_t episodes,
                std::size_t max_steps,
                std::mt19937 & prng)
{
    // the greedy agent: always takes the best known action
    auto agent = [&](const typename game_type::state_type & s,
                     const std::vector<typename game_type::action_type> & legal) {
        return greedy<game_type>(memory, s, legal, prng);
    };
    // play the games and average their returns
    double sum = 0;
    for (std::size_t i = 0; i < episodes; i++) {
        sum += play(game, agent, max_steps).total_reward;
    }
    return sum / static_cast<double>(episodes);
}
} // namespace games
#endif
