/**
 * `relearn/eligibility_traces.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `eligibility_traces`,
 * `sarsa_lambda` (`td_lambda` is the same struct, under the name of prediction), `watkins_q_lambda`,
 * `true_online_sarsa_lambda` (and so `true_online_td_lambda`), `lambda_return_record`,
 * `truncated_sarsa_lambda` (and so `truncated_td_lambda`) and `online_lambda_return`.
 * Including the header twice checks its include guard.
 */
#include "relearn/eligibility_traces.hpp"
#include "relearn/eligibility_traces.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template class relearn::eligibility_traces<S,A>;
template struct relearn::sarsa_lambda<S,A>;
template struct relearn::watkins_q_lambda<S,A>;
template struct relearn::true_online_sarsa_lambda<S,A>;
template class relearn::lambda_return_record<S,A>;
template struct relearn::truncated_sarsa_lambda<S,A>;
template struct relearn::online_lambda_return<S,A>;

int main()
{
    // the name of prediction gives the same struct
    relearn::td_lambda<S,A> td;
    relearn::true_online_td_lambda<S,A> true_online;
    relearn::truncated_td_lambda<S,A> truncated;
    return (td.traces.entries().empty() && true_online.traces.entries().empty() && truncated.n == 4) ? 0 : 1;
}
