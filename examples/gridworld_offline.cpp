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
    // explore until we discover a negative or positive reward!
    bool stop = false;
    // the markov_chain/episode we populate
    std::deque<relearn::link<S,A>> episode;

    // S_t (state now) is initially the root state
    std::cout << "starting exploration from: " << start.x 
              << "," << start.y << std::endl;
    grid curr = start;
    auto state_now = state(curr.R, curr);

    // explore while Reward isn't positive or negative
    // and keep populating the episode with states and actions
    while (!stop) 
    {
        // randomly decide on next grid - we map numbers to a direction
        auto result = rand_direction()(gen, w, curr);
        // find the reward at the current coordinates
        curr = result.second;
        // create the action using direction as trait
        auto action_now = action(result.first);
        // add the state to the episode
        episode.emplace_back(relearn::link<state,action>{state_now, action_now});
        // update current state to next state
        state_now = state(curr.R, curr);
        if (curr.R == -1 || curr.R == 1) {
            stop = true;
        }
        // print on screen
        std::cout << "coord: " << curr.x << "," 
                  << curr.y << " = " << curr.R << std::endl;
    }
    // Add the terminal state last
    // note: we don't have an empty action - we could use pointers instead...
    auto action_empty = action(direction({100}));
    episode.emplace_back(relearn::link<state,action>{state_now, action_empty});
    return episode;
}

/// Stay On-Policy and execute the action dictated
template <typename S, typename A>
void on_policy(const world & w, 
               relearn::policy<S,A> & policy_map,
               grid start)
{
    grid curr = start;
    std::cout << "starting from: " << curr.x << "," 
              << curr.y << " = " << curr.R << std::endl;
    auto state_t = S(curr.R, curr);
    for (;;) {
        // get the best policy for this state from the episode
        if (auto action = policy_map.best_action(state_t))
        {    
            // how to infer the next state
            switch (action->trait().dir) {
                case 0 : curr.y--;
                         break;
                case 1 : curr.x++;
                         break;
                case 2 : curr.y++;
                         break;
                case 3 : curr.x--;
                         break;
            }
            auto it = w.blocks.find(curr);
            if (it != w.blocks.end()) {
                curr = *it;
                // calculate our next state
                auto state_n = S(curr.R, curr);
                std::cout << "coord: " << curr.x << "," 
                          << curr.y << " = " << curr.R << std::endl;
                state_t = state_n;
                if (curr.R == -1.0 || curr.R == 1.0) {
                    break;
                }
            }
        }
    }
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
    auto learner = relearn::q_learning<state,action>{0.9, 0.9};
    for (int k = 0; k < 10; k++) {
        for (auto episode : episodes) {
            learner(episode, policies);
        }
    }

    // run on-policy and follow max Q policy action
    std::cout << "on-policy algorithm" << std::endl;
    on_policy(w, policies, start);
    return 0;
}
