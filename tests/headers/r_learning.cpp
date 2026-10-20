/**
 * `relearn/r_learning.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `r_learning` and `differential_sarsa`.
 * Including the header twice checks its include guard.
 */
#include "relearn/r_learning.hpp"
#include "relearn/r_learning.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::r_learning<S,A>;
template struct relearn::differential_sarsa<S,A>;

int main()
{
    return 0;
}
