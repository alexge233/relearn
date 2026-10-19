/**
 * `relearn/schedules.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of the schedules, the update counter and `step_size`.
 * Including the header twice checks its include guard.
 */
#include "relearn/schedules.hpp"
#include "relearn/schedules.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::polynomial_decay<>;
template struct relearn::exponential_decay<>;
template struct relearn::linear_decay<>;
template class relearn::visit_counter<S,A>;
template double relearn::step_size(const std::function<double(std::size_t)> &,
                                   double,
                                   relearn::visit_counter<S,A> &,
                                   const S &,
                                   const A &);

int main()
{
    return 0;
}
