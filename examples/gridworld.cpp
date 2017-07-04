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
#include <sstream>
#include <unordered_set>
#include <random>
#include <ctime>
#include <chrono>
#include <fstream>
#include <string>

#include "../src/relearn.hpp"

/**
 * A grid block is simply a coordinate (x,y)
 * A grid *may* have a reward R.
 * This struct functions as the block upon which the gridworld problem is based.
 * We also use this as the `state_trait` descriptor S for state<S,A> and action<S,A>
 */
struct grid
{
    unsigned int x = 0;
    unsigned int y = 0;
    double R = 0;
    bool occupied = false;

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
    std::unordered_set<grid> blocks;
};

using state = relearn::state<grid>;
using action = relearn::action<direction>;

///
/// load the gridworld from the text file
/// boundaries are `occupied` e.g., can't move into them
/// fire/danger blocks are marked with a reward -1
///
world populate()
{
    std::ifstream infile("../examples/gridworld.txt");
    world environment = {};
    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        unsigned int x;
        unsigned int y;
        double r;
        bool occupied;
        if (iss >> x >> y >> occupied >> r) {
            environment.blocks.insert({x, y, r, occupied});
        }
        else break;
    }
    return environment;
}

///
/// Decide on a stochastic (random) direction and return the next grid block
///
struct rand_direction
{
    std::pair<direction,grid> operator()(std::mt19937 & prng, 
                                         world gridworld, 
                                         grid current)
    {
        std::uniform_int_distribution<unsigned int> dist(0, 3);
        unsigned int x = current.x;
        unsigned int y = current.y;
        // randomly decide on next grid - we map numbers to a direction
        unsigned int d = dist(prng);
        switch (d) {
            case 0 : y--;
                     break;
            case 1 : x++;
                     break;
            case 2 : y++;
                     break;
            case 3 : x--;
                     break;
        }
        auto it = std::find_if(gridworld.blocks.begin(),
                               gridworld.blocks.end(),
                               [&](const auto b) {
                                   return b.x == x && b.y == y;
                               });
        if (it == gridworld.blocks.end()) {
            //std::cerr << "tried to move off the grid at: " << x << "," << y << std::endl;
            return rand_direction()(prng, gridworld, current);
        }
        if (it->occupied) {
            //std::cerr << "occupied block: " << x << "," << y << std::endl;
            return rand_direction()(prng, gridworld, current);
        }
        return std::make_pair(direction{d}, *it);
    }
};

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

///
/// Stay On-Policy and execute the action dictated
///
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

    // set shortcuts to state trait and action trait
    using state = relearn::state<grid>;
    using action = relearn::action<direction>;

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
