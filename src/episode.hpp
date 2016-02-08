#ifndef episode
#define episode
namespace relearn
{
/// episode holds a sequence of states and actions.
/// chose the type of container to store states and actions,
/// and then chose the type of algorithm to reinforce them.
///
template <typename container, template learner>
class episode
{
public:
    using state_pt = std::unique_ptr<state>;
    using action_pt = std::unique_ptr<action>;
    using const_iterator = container<state_pt>::const_iterator;

    // markov chain is series of states
    container<state_pt> states;
    container<action_ptr> actions;
    // episode reward
    float reward = 0.f;
    
    /// \brief reinforce \param policies
    /// \note requires that a reward has been set
    template <typename P>
    void reinforce(const P<policy> policies)
    {
        const learner & algorithm(policies);
        std::for_each(states.begin(),
                      states.end(),
                      algorithm);
    }
    // equality
    bool operator==(const episode &arg) const
    {
         return typeid(*this) == typeid(arg) 
                && is_equal(arg);   
    }
    // comparison
    bool operator<(const episode &arg) const
    {
        return compare(arg);
    }
    // iterator episode states
    const state_pt & operator[](const int i) const
    {
        return states[i];
    }
    // const iterator begin
    const_iterator begin() const
    {
        return states.begin();
    }
    // const iterator end
    const_iterator end() const
    {
        return states.end();
    }
    // episode size
    unsigned int size() const
    {
        return states.size();
    }
private:
    // equality based on member `states`
    virtual bool is_equal(const episode &arg) const
    {
        return std::equal(this->stats.begin(), 
                          this->states.end(),
                          arg.states().begin(),
                          states_equal);
    }
    // equality based on episode `rewards`
    virtual bool compare(const episode &arg) const
    {
         return reward < arg.reward;
    }
};
}
#endif /*episode*/
