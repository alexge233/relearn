/**
 * `relearn/policy_gradient.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `softmax_policy`,
 * `reinforce`, `reinforce_with_baseline`, `actor_critic`, `actor_critic_lambda` and `continuing_actor_critic`
 * (and the `fade_traces` they share).
 * Including the header twice checks its include guard.
 */
#include "relearn/policy_gradient.hpp"
#include "relearn/policy_gradient.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::softmax_policy<S,A>;
template struct relearn::reinforce<S,A>;
template struct relearn::reinforce_with_baseline<S,A>;
template struct relearn::actor_critic<S,A>;
template struct relearn::actor_critic_lambda<S,A>;
template struct relearn::continuing_actor_critic<S,A>;

int main()
{
    return 0;
}
