/**
 * `relearn/bandits.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `ucb` and `gradient_bandit`.
 * Including the header twice checks its include guard.
 */
#include "relearn/bandits.hpp"
#include "relearn/bandits.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::ucb<S,A>;
template struct relearn::gradient_bandit<S,A>;

int main()
{
    return 0;
}
