#ifndef GRIDWORLD_HPP
#define GRIDWORLD_HPP
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
 * This header contains the basic structures and operations
 * needed to demonstrate the Gridworld ecosystem in a simple
 * yet fully functional manner.
 *
 * 1. State space: GridWorld has 10x10 = 100 distinct states. 
 *    The start state is the top left cell. 
 *    The gray cells are walls and cannot be moved to.
 *
 * 2. Actions: The agent can choose from up to 4 actions to move around.
 *
 * 3. Rewards: The agent receives +1 reward when it is in 
 *    the center square (the one that shows R 1.0), 
 *    and -1 reward in a the boundary states (R -1.0 is shown for these). 
 *    The state with +1.0 reward is the goal state.
 */


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

/// load the gridworld from the text file
/// boundaries are `occupied` e.g., can't move into them
/// fire/danger blocks are marked with a reward -1
world populate()
{
    std::ifstream infile("../examples/gridworld.txt");
    assert(infile.good()); // Fails if the file cannot be found
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

/// Decide on a stochastic (random) direction and return the next grid block
struct rand_direction
{
    std::pair<direction,grid> operator()(std::mt19937 & prng, 
                                         world gridworld, 
                                         grid current)
    {
        assert(!gridworld.blocks.empty()); // Fails if world is empty
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
            return rand_direction()(prng, gridworld, current);
        }
        if (it->occupied) {
            return rand_direction()(prng, gridworld, current);
        }
        return std::make_pair(direction{d}, *it);
    }
};
#endif
