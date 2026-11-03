/**
 * `relearn/planning.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiations below compile every member function of `trajectory_sampling`,
 * `real_time_dynamic_programming`, `heuristic_search` and `rollout_planner`, and the templated ones are called.
 * Including the header twice checks its include guard.
 */
#include "relearn/planning.hpp"
#include "relearn/planning.hpp"

using S = relearn::state<int>;
using A = relearn::action<int>;

template struct relearn::trajectory_sampling<S,A>;
template struct relearn::real_time_dynamic_programming<S,A>;
template struct relearn::heuristic_search<S,A>;
template struct relearn::rollout_planner<S,A>;

int main()
{
    // one state, 0, whose only action ends the episode with a reward of 1
    relearn::finite_mdp<S,A> model;
    model.states = {S(0), S(1)};
    model.actions = [](const S & s) { return s.trait() == 0 ? std::vector<A>{A(0)} : std::vector<A>{}; };
    model.outcomes = [](const S &, const A &) { return std::vector<relearn::finite_mdp<S,A>::outcome>{{S(1), 1.0, 1.0}}; };
    std::mt19937 prng(0);
    relearn::trajectory_sampling<S,A> sampling;
    relearn::policy<S,A> q;
    sampling.plan(model, q, S(0), 2, prng);
    relearn::real_time_dynamic_programming<S,A> rtdp;
    rtdp.trial(model, S(0), prng);
    relearn::heuristic_search<S,A> search;
    relearn::rollout_planner<S,A> rollouts;
    const bool fine = q.value(S(0), A(0)) == 1 && rtdp.value(S(0)) == 1 && search.choose(model, S(0)) == A(0) &&
                      rollouts.choose(model, S(0), prng) == A(0) && relearn::sample_outcome(model, S(0), A(0), prng).reward == 1;
    return fine ? 0 : 1;
}
