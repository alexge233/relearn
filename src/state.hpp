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
    /// variadic constructor or initializer list of actions

    /// TODO: state owns **one to many** actions - we must be able to iterate them

    /// \brief add an action
    void operator<<(A arg);

    /// \brief state equality
    bool operator==(const state & arg) const;

    /// \brief descriptor used for hashing
    std::size_t operator()() const;
    
private:
    //  unique actions - immutable set, mutable objects = (actions must have next state?)
    //                  or can the objects be immutable too (actions will require next state beforehand)
    std::unordered_set<A> __actions__;
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
