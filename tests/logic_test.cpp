#include <string>
#include "relearn.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// TODO: create a LOGIC test, where we manually populate a determined list
//       of episodes (e.g., (0,0), (0,1), (0,2), (0,3), (0,4)
//       with their appropriate actions (e.g., N, E, W, S)
//       and a respective Reward
//       Repeat a few times (2-3 times is enough) with a positive and negative reward
//       then train the algorithm and policies,
//       and finally hard-core the policies and Q-Values to make sense
//       this way we can logically validate that the algorithm works fine
//       without issues or undefined behaviour
//
