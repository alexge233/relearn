#if USING_BOOST_SERIALIZATION
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/unordered_map.hpp>

template <class state_class>
struct state_wrapper : private state_class
{
    state_wrapper();
    operator state_class() const;
};

template <class action_class>
struct action_wrapper : private action_class
{
    action_wrapper();
    operator action_class() const;
};

template <class state_class>
struct cast_state
{
    state_wrapper<state_class> operator()(const state_class arg) const;
};

template <class action_class>
struct cast_action
{
    action_wrapper<action_class> operator()(const action_class arg) const;
};

template <class state_class> 
struct hasher<state_wrapper<state_class>>
{
    std::size_t operator()(const state_wrapper<state_class> & arg) const;
};

template <class action_class> 
struct hasher<action_wrapper<action_class>>
{
    std::size_t operator()(const action_wrapper<action_class> & arg) const;
};

template <class action_class,
         typename value_type>
struct hasher<std::unordered_map<action_class,value_type>>
{
    std::size_t operator()(const std::unordered_map<action_class,value_type> &arg) const;
};

template <class action_class,
          typename value_type>
struct hasher<std::unordered_map<action_wrapper<action_class>,value_type>>
{
    std::size_t operator()(const std::unordered_map<action_wrapper<action_class>,
                                                    value_type> &arg) const;
};


/// TODO: implement those methods above

#endif
