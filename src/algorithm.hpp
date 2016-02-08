#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP 
#include "includes"
#include "policy.hpp"
namespace relearn
{
/// Q-Learning
///
template <class container>
struct q_learning
{
    q_learning(const container<policy>& policies, 
               float alpha, 
               float gamma)
    : p(policies), a(alpha), g(gamma){}

    /// \brief update a policy
    /// \return Q(S[t],A[t]) + α(R[t+1] + γ*max(Q(S[t+1],A[t+1]) - Q(S[t],A[t])
    float operator()(const policy & Q) const
    {
        float maxQ = <container>find_max_policy(p)(Q.next);
        float R = Q.next->reward();
        return Q.value + a *(R + (g * maxQ) - Q.value);
    }

    const container<policy> p;
    const float a;
    const float g;
};

/// S.A.R.S.A. TD
///

/// Actor-Critic
///

/// Continous-R (?)
///

}
#endif /* ALGORITHM_HPP */
