#ifndef RELEARN_STATE_HPP
#define RELEARN_STATE_HPP
namespace relearn {
/**
 * \brief a state class
 * \class state
 * \version 0.1.0
 * \date 26-8-2016
 *
 * The state class owns *one to many* actions of type A
 * Those actions lead to next states, e.g., a tree structure
 */
template <typename A>
class state
{
public:
 
    /// construct with a reward (terminal state)
    state(float reward);   

    /// construct with [0...n] actions
    state(std::initializer_list<A> actions);

    /// construct with [0...n] actions and a reward (oxymoron)
    state(std::initializer_list<A> actions, float reward);

    /// \brief add an action - unique, no duplicates
    void operator<<(A arg);

    /// \brief state equality
    bool operator==(const state<A> & arg) const;

    /// \return unique hash
    std::size_t operator()() const;

    /// action const iterator
    typedef std::unordered_set<A>::const_iterator const_iter;

    /// \brief const iterator for actions: begin
    const_iter begin() const;

    /// \brief const iterator for actions: end 
    const_iter end() const;

    /// \return reward
    float reward() const;
    
private:
    //  unique actions - immutable set
    std::unordered_set<A> __actions__;
    // state reward
    float __reward__ = .0f;
};


/// \brief hash functor for state
template <> struct hash<state>
{
    std::size_t operator()(state const & arg) const
    {
        return arg()();;
    } 
};
}
#endif
