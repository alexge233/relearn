/**
 * `relearn/monte_carlo.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `monte_carlo`, `off_policy_monte_carlo`
 * and `off_policy_monte_carlo_prediction`.
 * Including the header twice checks its include guard.
 */
#include "relearn/monte_carlo.hpp"
#include "relearn/monte_carlo.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::monte_carlo<S,A>;
template struct relearn::off_policy_monte_carlo<S,A>;
template struct relearn::off_policy_monte_carlo_prediction<S,A>;

int main()
{
    return 0;
}
