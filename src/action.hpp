#ifndef RELEARN_ACTION_HPP
#define RELEARN_ACTION_HPP
namespace relearn {
/**
 * \brief an action class - servces as an example but may be used as base
 * \class action
 * \version 0.1.0
 * \date 26-8-2016
 */
template <typename S>
class action
{
public:
    /// \brief construct using \param next state
    action(S state); 

    /// \brief get next state
    const & S state() const;

    /// equality operator
    bool operator==(const action & arg) const;

    /// descriptor used for hashing
    std::size_t operator()() const;

private:
    /// next state
    S __next__;    
};


/// \brief hash functor for action
template <> struct hash<action>
{
    std::size_t operator()(action const & arg) const
    {
        return arg()();
    } 
};
}
#endif
