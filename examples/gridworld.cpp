/**
 * the Gridworld problem as described by Sutton and Barto.
 * This is a toy environment called Gridworld that is often used 
 * as a toy model in the Reinforcement Learning literature. 
 * In this particular case:
 *
 * 1. State space: GridWorld has 10x10 = 100 distinct states. 
 *    The start state is the top left cell. 
 *    The gray cells are walls and cannot be moved to.
 *
 * 2. Actions: The agent can choose from up to 4 actions to move around.
 *
 * 3. Environment Dynamics: GridWorld is deterministic, 
 *    leading to the same new state given each state and action.
 *    Non-deterministic environments nead a probabilistic approach.
 *
 * 4. Rewards: The agent receives +1 reward when it is in 
 *    the center square (the one that shows R 1.0), 
 *    and -1 reward in a the boundary states (R -1.0 is shown for these). 
 *    The state with +1.0 reward is the goal state.
 *
 * This is a deterministic, finite Markov Decision Process (MDP) 
 * and the goal is to find an agent policy that maximizes 
 * the future discounted reward.
 */
#include <iostream>
#include <unordered_set>
#include <random>
#include <ctime>
#include <chrono>

#include "../src/relearn.hpp"

/**
 * A grid block is simply a coordinate (x,y)
 * A grid *may* have a reward R.
 * This struct functions as the block upon which the gridworld problem is based.
 * We also use this as the `state_trait` descriptor S for state<S,A> and action<S,A>
 */
struct grid
{
    unsigned int x;
    unsigned int y;
    double R;

    bool operator==(const grid & arg) const
    {
        return (this->x == arg.x) && 
               (this->y == arg.y);
    }
};

/**
 * A move in the grid world is simply a number.
 * 0 for left, 1 for top, 2 for right, 3 for down.
 * We also use this as the `action_trait` descriptor A for state<S,A> and action<S,A>
 */
struct direction
{
    unsigned int dir;

    bool operator==(const direction & arg) const
    {
        return (this->dir == arg.dir);
    }
};

/**
 * Hash specialisations in the STD namespace for structs grid and direction.
 * Those are **required** because the underlying relearn library 
 * uses unordered_map and unordered_set, which use hashing functions for the classes
 * which are mapped/stored internally.
 */
namespace std 
{
template <> struct hash<grid>
{
    std::size_t operator()(grid const& arg) const 
    {
        std::size_t seed = 0;
        relearn::hash_combine(seed, arg.x);
        relearn::hash_combine(seed, arg.y);
        return seed;
    }
};
template <> struct hash<direction>
{
    std::size_t operator()(direction const& arg) const
    {
        std::size_t seed = 0;
        relearn::hash_combine(seed, arg.dir);
        return seed;
    }
};
}

/**
 * The gridworld struct simply contains the grid blocks.
 * Each block is uniquely identified by its coordinates.
 */
struct world
{
    const grid start;
    std::unordered_set<grid> blocks;
};

/**
 * populate a gridworld:
 * 1. only one block will have a positive reward
 * 2. all edge/boundary blocks will have a negative reward
 * 
 * You may change this, make it more complex, larger or "interesting" ;-)
 */
world populate()
{
    unsigned int height = 5;
    unsigned int width  = 5;

    // world - start at 1,1
    world environment = {{ 1, 1, .0}};

    // pick one goal/end block
    grid goal = { 2, 3, 1};
    environment.blocks.insert(goal);
    std::cout << "goal is at: " << 2 << "," << 3 << std::endl;

    // populate the remaining grids: -1 for edges, zero for all others
    // iterate height first, width second and add grid blocks, 
    for (unsigned int i = 0; i < width; i++) {
        for (unsigned int k = 0; k < height; k++) {
            if (i == 0 || k == 0) {
                environment.blocks.insert({i, k, -1.});
            }
            else if (i == width - 1 || k == height - 1) {
                environment.blocks.insert({i, k, -1.});
            } 
            else {
                environment.blocks.insert({i, k, 0.});
            }
        }
    }
    return environment;
}

/**
 * Exploration technique is based on Monte-Carlo, e.g.: stochastic search.
 * The `agent` will randomly search the world experiencing different blocks.
 * The agent will internally map its experience using the State/Action pairs.
 */
template <typename S, typename A>
relearn::markov_chain<S,A> explore(
                                    const world & w,
                                    std::mt19937 & gen
                                  )
{
    using state = relearn::state<grid>;
    using action = relearn::action<direction>;

    std::uniform_int_distribution<unsigned int> dist(0, 3);

    bool stop = false;
    relearn::markov_chain<S,A> episode;

    // S_t (state now) is initially the root state
    grid curr  = w.start;
    auto state_now = state(curr.R, curr);

    // explore while Reward isn't positive or negative
    // and keep populating the episode with states and actions
    while (!stop) 
    {
        // randomly decide on next grid - we map numbers to a direction
        // and at the same time infer the next state
        unsigned int d = dist(gen);
        switch (d) {
            case 0 : curr.y--;
                     break;
            case 1 : curr.x++;
                     break;
            case 2 : curr.y++;
                     break;
            case 3 : curr.x--;
                     break;
        }
        // find the reward at the current coordinates
        auto it = w.blocks.find(curr);
        if (it != w.blocks.end()) {
            curr = *it;
            std::cout << "coord: " << curr.x << "," 
                      << curr.y << " = " << curr.R << std::endl;
            // create the action using direction as trait
            auto action_now = action(direction({d}));
            // add the state to the episode
            episode.emplace_back(relearn::link<state,action>{
                                  std::make_shared<state>(state_now),
                                  std::make_shared<action>(action_now)});
            // update current state to next state
            state_now = state(it->R, *it);
            if (it->R == -1 || it->R == 1) {
                stop = true;
            }
        }
        else {
            throw std::runtime_error("illegal block");
        }
    }
    // Add the terminal state last
    episode.emplace_back(relearn::link<state,action>{
                                  std::make_shared<state>(state_now),
                                  nullptr});
    return episode;
}

template <typename S, typename A>
void on_policy(const world & w, relearn::policy<S,A> & policy_map)
{
    grid curr = w.start;
    std::cout << "starting from: " << curr.x << "," 
              << curr.y << " = " << curr.R << std::endl;
    auto state_t = S(curr.R, curr);
    for (;;) {
        // get the best policy for this state from the episode
        if (auto action = policy_map.best_action(state_t)) {
            
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
            std::cout << "best action: " << action->trait().dir << std::endl;
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
            else {
                throw std::runtime_error("unknown grid");
            }       
        }
        else {
            throw std::runtime_error("no best action");
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
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));

    // set shortcuts to state trait and action trait
    using state = relearn::state<grid>;
    using action = relearn::action<direction>;

    // create the world and populate it randomly
    world w = populate();
    relearn::policy<state,action> policies;
    std::vector<relearn::markov_chain<state,action>> episodes;

    for (;;) {
        // explore the grid world randomly - this produces an episode
        auto episode = explore<state,action>(w, gen);
        episodes.push_back(episode);

        // check solution has been found
        auto it = std::find_if(episode.begin(), episode.end(),
                 [&](const auto & link) {
                    return link.state_t->reward() == 1;
                 });
        if (it != episode.end()) {
            break;
        }
    }

    // use Q-learning algorithm to update the episode's policies
    auto learner = relearn::q_learning<state,action>(0.9, 0.1);
    for (int k = 0; k < 10; k++) {
        for (auto episode : episodes) {
            learner(episode, policies);
        }
    }

    // run on-policy and follow maxQ
    on_policy(w, policies);

    return 0;
}

