/**
 * `relearn/q_learning.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `q_learning` and `q_probabilistic`.
 * Including the header twice checks its include guard.
 */
#include "relearn/q_learning.hpp"
#include "relearn/q_learning.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::q_learning<S,A>;
template struct relearn::q_probabilistic<S,A>;

int main()
{
    return 0;
}
