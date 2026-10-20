/**
 * `relearn/prioritized_sweeping.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiation below compiles every member function of `prioritized_sweeping`.
 * Including the header twice checks its include guard.
 */
#include "relearn/prioritized_sweeping.hpp"
#include "relearn/prioritized_sweeping.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::prioritized_sweeping<S,A>;

int main()
{
    return 0;
}
