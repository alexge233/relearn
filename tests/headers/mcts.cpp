/**
 * `relearn/mcts.hpp` compiles on its own, with nothing included before it.
 *
 * Templates are only compiled when they are used, so including the header is not enough:
 * the explicit instantiation below compiles every member function of `mcts`, with a model
 * of the smallest shape `mcts` needs. Including the header twice checks its include guard.
 */
#include "relearn/mcts.hpp"
#include "relearn/mcts.hpp"

/// the outcome of a step of the model below
struct outcome
{
    int next;
    double reward;
    bool terminated;
};

/// a model with one action, which ends the game
struct model
{
    using state_type  = int;
    using action_type = int;

    std::vector<int> actions(const int &) const
    {
        return {0};
    }

    outcome step(const int & s,
                 const int &) const
    {
        return {s + 1, 1.0, true};
    }

    unsigned int player(const int &) const
    {
        return 0;
    }
};

template struct relearn::mcts<model>;

int main()
{
    return 0;
}
