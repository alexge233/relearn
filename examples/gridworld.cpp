/**
 * the Gridworld problem as described by Sutton and Barto.
 * This is a toy environment called Gridworld that is often used as a toy model in the Reinforcement Learning literature. 
 * In this particular case:
 *
 * 1. State space: GridWorld has 10x10 = 100 distinct states. 
 *    The start state is the top left cell. 
 *    The gray cells are walls and cannot be moved to.
 *
 * 2. Actions: The agent can choose from up to 4 actions to move around.
 *
 * 3. Environment Dynamics: GridWorld is deterministic, leading to the same new state given each state and action.
 *
 * 4. Rewards: The agent receives +1 reward when it is in the center square (the one that shows R 1.0), 
 *    and -1 reward in a few states (R -1.0 is shown for these). 
 *    The state with +1.0 reward is the goal state.
 *
 * In other words, this is a deterministic, finite Markov Decision Process (MDP) and as always the goal is to find an agent policy
 * (shown here by arrows) that maximizes the future discounted reward.
 */
#include <iostream>
#include <unordered_set>
#include <random>
#include "relearn.hpp"
#include "action_state.hpp"

/**
 * A grid block is simply a coordinate (x,y)
 * A grid *may* have a reward R or it may be blocked (can't move there)
 * This functions as the block upon which the gridworld problem is based.
 * We also use this as the `trait` descriptor S for state<S,A> and action<S,A>
 */
struct grid
{
    unsigned int x;
    unsigned int y;
    float R;

    bool operator==(const grid & arg) const
    {
        return (this->x == arg.x) && 
               (this->y == arg.y);
    }
};

/**
 * A move in the grid world is simply a number.
 * 0 for left, 1 for top, 2 for right, 3 for down.
 * We also use this as the `trait` descriptor A for state<S,A> and action<S,A>
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
 * Those are **required** because the underlying relearn library uses unordered_map and unordered_set
 */
namespace std {
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
 * 2. all edge blocks will have a negative reward
 */
world populate(unsigned int height, unsigned int width)
{
    std::default_random_engine eng((std::random_device())());    
    std::uniform_real_distribution<> dist(0, 10);

    // world - start at 1,1
    world environment = {{ 1, 1, .0f}};

    // pick one random block which will be the goal
    unsigned int x = dist(eng);
    unsigned int y = dist(eng);
    grid goal = { x, y, 1.f};
    environment.blocks.insert(goal);

    // iterate height first, width second and add grid blocks, 
    for (unsigned int i = 0; i < width; i++) {
        for (unsigned int k = 0; k < height; k++) {
            if (i == 0 || k == 0) {
                environment.blocks.insert({i, k, -1.f});
            }
            else if (i == width - 1 || k == height - 1) {
                environment.blocks.insert({i, k, -1.f});
            } 
            else {
                environment.blocks.insert({i, k, 0.f});
            }
        }
    }
    return environment;
}

/**
 * Exploration technique is based on Monte-Carlo, e.g.: stochastic/random search.
 * The `agent` will normally search the world experiencing different grids.
 * On-Line mode will update as exploring, whilst Off-Line will update after exploring (?)
 * The agent will internally map its experience using the State/Action pairs, and recording Policies for each pair.
 */
template <typename S, typename A>
void explore(const world & w, relearn::episode<S, A> & e)
{
    using state = relearn::state<grid, direction>;
    using action = relearn::action<grid, direction>;

    std::default_random_engine eng((std::random_device())());    
    std::uniform_real_distribution<> dist(0, 3);

    auto start = w.start;
    grid curr  = start;
    float R    = 0;
    bool stop = false;

    // TODO: get root state from episode then get action
    //       and add action to state (root or current)
    //       thereby populating the episode

    // explore while Reward isn't positive or negative
    // and keep populating the episode with states and actions
    do {
        // randomly decide on next grid - we map numbers to a direction
        // and at the same time infer the next state
        unsigned int d = dist(eng);
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
            std::cout << "coord: " << curr.x << "," << curr.y << " = " << it->R << std::endl;
            R = it->R;
        }

        // stop populating once we've reached a negative or positive reward
        stop = (R < 0 || R == 1) ? true : false;

        // create next state
        auto s_next = state(R, curr);

        // create the action that was taken using the direction as trait and the next state
        auto a_t    = action(s_next, direction({d}));

        // TODO: 
    }
    while (!stop);

    // TODO: update episode policies
}

/**
 * Gridworld example main function.
 * The agent will begin with stochastic exploration, and at every terminal state will update its experience.
 * It will then reset back to the start of the episode and re-do the entire process for 100 iterations.
 * take as argc/argv the iteration count?
 */
int main()
{
    // create the world and populate it randomly
    world w = populate(10, 10);

    // set shortcuts to state trait and action trait
    using state = relearn::state<grid, direction>;
    using action = relearn::action<grid, direction>;

    // create an episode using the starting grid as the root state
    auto episode = relearn::episode<state, action>(state({w.start.R, w.start}));

    // explore once (repeat until?)
    explore(w, episode);

    // TODO: update values using Q-learning
    // TODO: repeat explore - update for 100 times.

    return 0;
}

