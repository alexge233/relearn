/**
 * `relearn/dynamic_programming.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `finite_mdp` and `dynamic_programming`.
 * Including the header twice checks its include guard.
 */
#include "relearn/dynamic_programming.hpp"
#include "relearn/dynamic_programming.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::finite_mdp<S,A>;
template struct relearn::dynamic_programming<S,A>;

int main()
{
    return 0;
}
