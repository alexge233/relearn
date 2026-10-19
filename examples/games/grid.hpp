#ifndef RELEARN_GAMES_GRID_HPP
#define RELEARN_GAMES_GRID_HPP
/**
 * Positions and moves shared by the grid games.
 * Column x grows to the right, row y grows downwards; (0, 0) is the top-left cell.
 */
#include "../../src/relearn.hpp"
#include <functional>

namespace games
{
/**
 * @struct position
 * @brief a cell of a grid, given by its column `x` and its row `y`
 */
struct position
{
    int x = 0;
    int y = 0;

    /// @brief two positions are equal when they are the same cell
    bool operator==(const position & rhs) const
    {
        return (this->x == rhs.x) &&
               (this->y == rhs.y);
    }

    /// @brief two positions differ when they are different cells
    bool operator!=(const position & rhs) const
    {
        return !(*this == rhs);
    }
};

/// the four moves of the grid games
enum : unsigned int
{
    up    = 0,
    right = 1,
    down  = 2,
    left  = 3
};

/**
 * @brief move one cell from @param from in direction @param move
 * @param width and @param height are the size of the grid
 * @return the cell reached: moving into the border of the grid stays put
 */
inline position moved(const position & from,
                      unsigned int move,
                      int width,
                      int height)
{
    position to = from;
    // take one step in the direction of the move
    switch (move) {
        case up:
            to.y--;
            break;
        case right:
            to.x++;
            break;
        case down:
            to.y++;
            break;
        case left:
            to.x--;
            break;
        default:
            break;
    }
    // a step out of the grid is undone: the agent bumps into the border
    if (to.x < 0) {
        to.x = 0;
    }
    if (to.y < 0) {
        to.y = 0;
    }
    if (to.x >= width) {
        to.x = width - 1;
    }
    if (to.y >= height) {
        to.y = height - 1;
    }
    return to;
}
} // namespace games

/**
 * Hash specialisation in the STD namespace for struct position.
 * It is **required** because relearn stores states in unordered maps,
 * which use hashing functions for the classes they store.
 */
namespace std
{
template <> struct hash<games::position>
{
    std::size_t operator()(const games::position & p) const
    {
        std::size_t seed = 0;
        relearn::hash_combine(seed, p.x);
        relearn::hash_combine(seed, p.y);
        return seed;
    }
};
}
#endif
