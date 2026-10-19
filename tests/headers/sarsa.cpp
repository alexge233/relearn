/**
 * `relearn/sarsa.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `sarsa` and `expected_sarsa`.
 * Including the header twice checks its include guard.
 */
#include "relearn/sarsa.hpp"
#include "relearn/sarsa.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::sarsa<S,A>;
template struct relearn::expected_sarsa<S,A>;

int main()
{
    return 0;
}
