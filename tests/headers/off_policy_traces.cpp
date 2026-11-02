/**
 * `relearn/off_policy_traces.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `off_policy_td_lambda`,
 * `off_policy_expected_sarsa_lambda` and `tree_backup_lambda`.
 * Including the header twice checks its include guard.
 */
#include "relearn/off_policy_traces.hpp"
#include "relearn/off_policy_traces.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::off_policy_td_lambda<S,A>;
template struct relearn::off_policy_expected_sarsa_lambda<S,A>;
template struct relearn::tree_backup_lambda<S,A>;

int main()
{
    relearn::tree_backup_lambda<S,A> tree;
    relearn::off_policy_td_lambda<S,A> td;
    return (tree.traces.entries().empty() && td.value(S(0)) == 0) ? 0 : 1;
}
