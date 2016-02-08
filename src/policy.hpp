#ifndef policy
#define policy
#include "includes.ihh"
namespace relearn
{
/// Policy Q(St,At) used for Learning
/// struct holds polymorphic pointers to pairs <State,Action>
///
struct policy
{
    // St,At
    const std::pair<state*, action*> pair;
    // St+1
    const state * next;
    // Q(St,At)
    float value = 0.f;

    // construct using a state and action
    policy(const state *s, const action *a, const state *next_s)
    : pair.first(s), pair.second(a), next(next_s)
    {}
    // equality based on state,action and next_state
    bool operator==(const policy &arg) const
    {
        return *pair.first == *arg.pair.first &&
               *pair.second == *arg.pair.second &&
               *next == *arg.next;
    }
    // sorting based only on value
    bool operator<(const policy &arg) const
    {
        return value < arg.value;
    }
};
/// find the max policy (of the next state)
template <class container>
struct find_max_policy
{
    find_max_policy(const container<policy>& policies)
    : p(policies){}

    float operator()(const state* arg) const
    {
        float maxQ = 0.f;
        std::for_each(p.begin(), p.end(),
                      [&](const policy & rhs)->void
                      {
                        if (*rhs.pair.first == *Q.next)
                            if (rhs.value > maxQ)
                                maxQ = rhs.value;
                      });
        return maxQ;
    }

    const container<policy> & p;
};
}
#endif /*policy*/
