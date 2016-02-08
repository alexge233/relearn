#ifndef state
#define state
namespace relearn
{
/// state class interface
///
class state 
{
public:
    // destructor
    virtual ~state();
    // equality
    bool operator==(const state & arg) const
    {
        return typeid(*this) == typeid(arg) 
               && is_equal(arg);
    }
    // comparison
    bool operator<(const state & arg) const
    {
        return compare(arg);
    }
    // get reward of this state
    float reward() const = 0;

private:
    // states equality
    bool is_equal(const state &) const = 0;
    // compare (sort)
    bool compare(const state &) const = 0;
};
// equality predicate
struct states_equal
{
    bool operator()(const std::unique_ptr<state> & lhs,
                    const std::unique_ptr<state> & rhs)
    {
        return (*lhs == *rhs);
    }
};
}
#endif /*state*/
