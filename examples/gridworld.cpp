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
#include <unordered_set>
#include <random>
#include "relearn.hpp"
/**
 * A grid block is simply a coordinate (x,y)
 * A grid *may* have a reward R or it may be blocked (can't move there)
 * This functions as the block upon which the gridworld problem is based.
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

template <class T> struct my_hash;
template<> struct my_hash<grid>
{
    std::size_t operator()(grid const& arg) const 
    {
        std::size_t seed = 0;
        relearn::hash_combine(seed, arg.x);
        relearn::hash_combine(seed, arg.y);
        return seed;
    }
};

/**
 * The gridworld struct simply contains the grid blocks.
 * Each block is uniquely identified by its coordinates.
 */
struct world
{
    std::unordered_set<grid, my_hash<grid>> blocks;

    // TODO: output as plantUML so that we can visualize the grid-world?
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

    // pick one random block which will be the goal
    unsigned int x = dist(eng);
    unsigned int y = dist(eng);
    grid goal = { x, y, 1.f};

    // world - add goal
    world environment;
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
template <typename S, typename A, typename N>
void explore(
              const world & w, 
              relearn::episode<S,A,N> & e
            )
{
    // TODO: run random actions, observe new state, update episode and append states/actions
    //       avoid taking same actions, this is an exploration ala Monte Carlo
}


/**
 * Gridworld example main function.
 * The agent will begin with stochastic exploration, and at every terminal state will update its experience.
 * It will then reset back to the start of the episode and re-do the entire process for 100 iterations.
 */
int main()
{
    world w = populate(10, 10);

    // TODO: create a main loop: explore, then update, explore then update
    //       repeat for 100 times, then serialize world as plantUML and visualise it online
    //       so that the actual example may be used as a paradigm.

    return 0;
}

