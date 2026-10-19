/**
 * `relearn/double_q_learning.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `double_q_learning`.
 * Including the header twice checks its include guard.
 */
#include "relearn/double_q_learning.hpp"
#include "relearn/double_q_learning.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::double_q_learning<S,A>;

int main()
{
    return 0;
}
