/**
 * `relearn/n_step.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiation below compiles every member function of `n_step_sarsa`
 * (`n_step_td` is the same struct, under the name of prediction).
 * Including the header twice checks its include guard.
 */
#include "relearn/n_step.hpp"
#include "relearn/n_step.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::n_step_sarsa<S,A>;

int main()
{
    // the name of prediction gives the same struct
    relearn::n_step_td<S,A> td;
    return td.n == 1 ? 0 : 1;
}
