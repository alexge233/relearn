#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unordered_map.hpp>

template <class state_class>
struct state_serial : public state_class
{
    state_serial();
    state_serial(state_class);
    operator state_class();
};

template <class action_class>
struct action_serial : public action_class
{
    action_serial();
    action_serial(action_class);
    operator action_class();
};

/*
 * Template serialization implementation
 */
template <class state_class>
state_serial<state_class>::state_serial()
: state_class()
{}

template <class state_class>
state_serial<state_class>::state_serial(state_class arg)
: state_class(arg)
{}

template <class state_class>
state_serial<state_class>::operator state_class() 
{
    return static_cast<state_class>(*this);
}

template <class action_class>
action_serial<action_class>::action_serial()
: action_class()
{}

template <class action_class>
action_serial<action_class>::action_serial(action_class arg)
: action_class(arg)
{}

template <class action_class>
action_serial<action_class>::operator action_class() 
{
    return static_cast<action_class>(*this);
}
