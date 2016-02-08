#ifndef action
#define action
namespace relearn
{
/// action interface
///
class action
{
public:
    // destructor
    virtual ~action();

    // equality
    bool operator==(const action &arg) const
    {
        return typeid(*this) == typeid(arg) 
               && is_equal(arg);
    }
    // compare
    bool operator<(const action &arg) const
    {
        return compare(arg);
    }

private:
    // are two actions equal
    bool is_equal(const action &) const = 0;
    // comparison
    bool compare(const action &) const = 0;
};
/// equality predicate
struct actions_equal
{
    bool operator()(const std::unique_ptr<action> & lhs,
                    const std::unique_ptr<action> & rhs)
    {
        return (*lhs == *rhs);
    }
};
}
#endif /*action*/
