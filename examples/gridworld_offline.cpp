/**
 * the Gridworld problem as described by Sutton and Barto.
 * This is a toy environment called Gridworld that is often used 
 * as a toy model in the Reinforcement Learning literature. 
 * In this particular case:
 *
 * Environment Dynamics: GridWorld is deterministic, 
 * leading to the same new state given each state and action.
 * This is a deterministic, finite Markov Decision Process (MDP) 
 * and the goal is to find an agent policy that maximizes 
 * the future discounted reward.
 *
 * This version of the Gridworld example uses off-line on-policy decision-making.
 * What that means is that as the agent moves, it only explores the environment.
 * It doens't learn anything until it has finished exploring the environment
 * and has discovered the "goal" state.
 *
 * Due to the nature of PRNG (pseudo-random number generator) this
 * version can get stuck into repeating the same actions over and over again,
 * therefore if it is running for longer than a minute, feel free to CTRL-C it.
 */
#include "gridworld_header.hpp"
// set shortcuts to state trait and action trait
using state = relearn::state<grid>;
using action = relearn::action<direction>;

/**
 * Exploration technique is based on Monte-Carlo, e.g.: stochastic search.
 * The `agent` will randomly search the world experiencing different blocks.
 * The agent will internally map its experience using the State/Action pairs.
 */
template <typename S, 
          typename A>
std::deque<relearn::link<S,A>> explore(const world & w,
                                       std::mt19937 & gen,
                                       grid start)
{
    // the markov_chain/episode we populate
    std::deque<relearn::link<S,A>> episode;

    std::cout << "starting exploration from: " << start.x 
              << "," << start.y << std::endl;
    grid curr = start;

    // explore until a positive or negative reward ends the episode
    while (!terminal(curr)) 
    {
        // randomly decide on next grid - we map numbers to a direction
        auto result = rand_direction()(gen, w, curr);
        // record (s_t, a_t) with the reward of the block we moved into
        episode.emplace_back(relearn::link<S,A>{S(result.second.R, curr), A(result.first)});
        curr = result.second;
        std::cout << "coord: " << curr.x << "," 
                  << curr.y << " = " << curr.R << std::endl;
    }
    return episode;
}

/**
 * Stay On-Policy and execute the action dictated
 * @return true if the goal was reached
 */
template <typename S, typename A>
bool on_policy(const world & w, 
               const relearn::policy<S,A> & policy_map,
               grid start)
{
    // a policy that goes round in circles is stopped
    const unsigned int max_steps = 100;
    grid curr = start;
    std::cout << "starting from: " << curr.x << "," 
              << curr.y << " = " << curr.R << std::endl;
    for (unsigned int step = 0; step < max_steps; step++) {
        // get the best policy for this state
        auto action = policy_map.best_action(S(curr));
        if (!action) {
            std::cout << "no policy for " << curr.x << "," << curr.y << std::endl;
            return false;
        }
        const grid * next = w.move(curr, action->trait().dir);
        if (!next) {
            std::cout << "policy leads into a wall at " << curr.x << "," << curr.y << std::endl;
            return false;
        }
        curr = *next;
        std::cout << "coord: " << curr.x << "," 
                  << curr.y << " = " << curr.R << std::endl;
        if (terminal(curr)) {
            return curr.R == 1.0;
        }
    }
    std::cout << "policy did not reach the goal in " << max_steps << " steps" << std::endl;
    return false;
}

/**
 * Gridworld example main function.
 * The agent will begin with stochastic exploration, 
 * and at every terminal state will update its experience.
 * It will then reset back to the start of the episode and 
 * re-do the entire process for 100 iterations.
 * 
 * After that, it will follow the learnt policies, and
 * perform actions of high value only.
 * This in effect will result in the agent learning the
 * environment and going straight for the target grid.
 */
int main()
{
    std::mt19937 gen(static_cast<std::size_t>(
        std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count()));

    // create the world and populate it randomly
    world w = populate();
    // hardcoded start grid
    grid start = {1, 8, 0};

    // store policies and episodes - we'll use policies later 
    relearn::policy<state,action> policies;
    std::vector<std::deque<relearn::link<state,action>>> episodes;

    // explore the grid world randomly - this produces an episode
    for (;;) {
        auto episode = explore<state,action>(w, gen, start);
        episodes.push_back(episode);
        // check solution has been found (e.g., positive reward!)
        auto it = std::find_if(episode.begin(), episode.end(),
                 [&](const auto & link) {
                    return link.state.reward() == 1.0;
                 });
        // we haz the terminal state - stop exploring
        // do note however that this may result in non-optimal policies
        if (it != episode.end()) break;
    }

    // use Q-learning algorithm to update the episode's policies
    // each pass over an episode moves the reward only one step back towards
    // the start, and random episodes can be long: replay them many times
    auto learner = relearn::q_learning<state,action>{0.9, 0.9};
    for (int k = 0; k < 200; k++) {
        for (const auto & episode : episodes) {
            learner(episode, policies);
        }
    }

    // run on-policy and follow max Q policy action
    std::cout << "on-policy algorithm" << std::endl;
    return on_policy<state,action>(w, policies, start) ? EXIT_SUCCESS : EXIT_FAILURE;
}
