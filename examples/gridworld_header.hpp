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
#include <vector>
#include <algorithm>
#include <cstdlib>
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
 * 3. Rewards: The agent receives +1 reward when it moves into
 *    the goal block (R 1.0), and -1 when it moves into a fire block (R -1.0).
 *    Both end the episode.
 *
 * Episodes follow relearn's reward convention: the link (s_t, a_t) carries
 * the reward received *after* taking a_t in s_t, i.e. the reward of the block
 * the agent moves into.
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

    /// @return the block at (@param x, @param y), or nullptr if there is none
    const grid * at(int x, int y) const
    {
        if (x < 0 || y < 0) {
            return nullptr;
        }
        grid key;
        key.x = static_cast<unsigned int>(x);
        key.y = static_cast<unsigned int>(y);
        auto it = blocks.find(key);
        return it != blocks.end() ? &(*it) : nullptr;
    }

    /**
     * @return the block reached by moving from @param from in direction @param dir
     *         (0 up, 1 right, 2 down, 3 left), or nullptr if it is a wall or
     *         outside the world
     */
    const grid * move(const grid & from, unsigned int dir) const
    {
        int x = static_cast<int>(from.x);
        int y = static_cast<int>(from.y);
        switch (dir) {
            case 0 : y--; break;
            case 1 : x++; break;
            case 2 : y++; break;
            case 3 : x--; break;
            default: return nullptr;
        }
        const grid * next = at(x, y);
        return (next && !next->occupied) ? next : nullptr;
    }
};

/// @return true if @param block ends an episode (the goal or a fire block)
inline bool terminal(const grid & block)
{
    return block.R == 1.0 || block.R == -1.0;
}

// the world file: CMake passes its absolute path, so the examples run from any directory
#ifndef GRIDWORLD_FILE
#define GRIDWORLD_FILE "../examples/gridworld.txt"
#endif

/// load the gridworld from the text file
/// boundaries are `occupied` e.g., can't move into them
/// fire/danger blocks are marked with a reward -1
world populate()
{
    std::ifstream infile(GRIDWORLD_FILE);
    if (!infile.good()) {
        // not an `assert`: asserts are compiled out in Release builds
        std::cerr << "cannot open " << GRIDWORLD_FILE << std::endl;
        std::exit(EXIT_FAILURE);
    }
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
    if (environment.blocks.empty()) {
        std::cerr << "no blocks found in " << GRIDWORLD_FILE << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return environment;
}

/// Decide on a stochastic (random) direction and return the next grid block
struct rand_direction
{
    std::pair<direction,grid> operator()(std::mt19937 & prng, 
                                         const world & gridworld, 
                                         const grid & current) const
    {
        // try the 4 directions in random order, and take the first free block
        std::vector<unsigned int> dirs = {0, 1, 2, 3};
        std::shuffle(dirs.begin(), dirs.end(), prng);
        for (unsigned int d : dirs) {
            if (const grid * next = gridworld.move(current, d)) {
                return std::make_pair(direction{d}, *next);
            }
        }
        std::cerr << "block " << current.x << "," << current.y 
                  << " is surrounded by walls" << std::endl;
        std::exit(EXIT_FAILURE);
    }
};
#endif
