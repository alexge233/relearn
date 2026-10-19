/**
 * `relearn/core.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `state`, `action`, `link` and `policy`.
 * Including the header twice checks its include guard.
 */
#include "relearn/core.hpp"
#include "relearn/core.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template class relearn::state<int>;
template class relearn::action<int>;
template struct relearn::link<S,A>;
template class relearn::policy<S,A>;

int main()
{
    return 0;
}
