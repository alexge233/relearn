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
    std::cout << "goal is at: " << x << "," << y << std::endl;

    // populate the remaining grids: -1 for edges, zero for all others
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
 * Exploration technique is based on Monte-Carlo, e.g.: stochastic search.
 * The `agent` will randomly search the world experiencing different blocks.
 * The agent will internally map its experience using the State/Action pairs.
 */
template <typename S, typename A>
void explore(const world &w, relearn::episode<S,A> &e)
{
    using state = relearn::state<grid, direction>;
    using action = relearn::action<grid, direction>;

    std::default_random_engine eng((std::random_device())());    
    std::uniform_real_distribution<> dist(0, 4);

    bool stop = false;

    // S_t (state now) is initially the root state
    auto state_now = e.root();
    grid curr  = w.start;

    // explore while Reward isn't positive or negative
    // and keep populating the episode with states and actions
    while (!stop) 
    {
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
            curr = *it;

            // create next state - BUG who owns the state? action keeps a reference!!!
            auto state_next = state(it->R, *it);
            // create the action using the direction as trait, and the next state
            auto action_now = action(state_next, direction({d}));

            if (state_now == state_next) {
                throw std::runtime_error("infinite loop");
            }

            // add the state tp the episode
            e.set_state(state_now, action_now);

            // update current state
            state_now = state_next;

            if (it->R == -1 || it->R == 1) {
                std::cout << "coord: " << it->x << "," 
                          << it->y 
                          << " = " 
                          << it->R << std::endl;
                stop = true;
            }
        }
        else {
            throw std::runtime_error("illegal block");
        }
    }
    e.set_state(state_now);
}

template <typename S, typename A>
void on_policy(const world &w, relearn::episode<S,A> &e)
{
    // get root state = current state
    auto state_t = e.root();
    grid location = state_t.trait();
    float R = location.R;

    std::cout << "starting from: " << location.x << "," 
              << location.y << " = " << location.R << std::endl;

    while (R != -1 && R != 1) {
        // get the best policy for this state from the episode
        auto action = e.best_policy(state_t);
        if (action) {
            // get the next state - print on screen
            auto state_n = action->next();

            location = state_n.trait();
            std::cout << "coord: " << location.x << "," 
                      << location.y << " = " << location.R << std::endl;
            R = location.R;
            state_t = state_n;
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
    // create the world and populate it randomly
    world w = populate(5, 5);

    for (int i = 0; i < 1; i++) {
        // set shortcuts to state trait and action trait
        using state = relearn::state<grid, direction>;
        using action = relearn::action<grid, direction>;

        // create an episode using the starting grid as the root state
        auto episode = relearn::episode<state,action>();

        // explore the grid world randomly
        explore(w, episode);

        // use Q-learning algorithm to update the episode's policies
        relearn::q_learning<state,action>()(episode, 0.7, 0.1);
    }

    /*
    // run on-policy and follow maxQ
    for (int i = 0; i < 10; i++) {
        on_policy(w, episode);
    }
    */

    return 0;
}

