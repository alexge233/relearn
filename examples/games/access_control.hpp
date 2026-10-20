#ifndef RELEARN_GAMES_ACCESS_CONTROL_HPP
#define RELEARN_GAMES_ACCESS_CONTROL_HPP
/**
 * The access-control queuing task, as in Sutton & Barto, 1st edition Example 6.7
 * (2nd edition Example 10.2): a task that never ends.
 *
 *     queue of customers  -->  [ accept or reject ]  -->  10 servers
 *     ... 4 1 8 2 [8]                                     # # . # . . # . . .
 *
 * - There are `servers` servers (10 in the book). Customers wait in a queue which never
 *   empties; each has a priority of 1, 2, 4 or 8, all equally likely.
 * - At each step the customer at the head of the queue is accepted or rejected.
 *   Accepted, it takes a free server and pays its priority as the reward;
 *   rejected (or accepted when no server is free), it leaves and pays 0.
 * - Then each busy server, including one just taken, becomes free with probability
 *   `free_probability` (0.06 in the book), and the next customer comes to the head.
 * - A state is the number of free servers and the priority of the customer at the head.
 *   With no free server, the only action is to reject.
 * - It never ends: there is no final state, and no episodes. `reset()` does not start again,
 *   it returns the state the queue is in, so `games::play_online` can be called again and
 *   again, each call carrying on where the last one stopped. The queue starts (when the game
 *   is created) with every server free.
 * - `average_reward(policy)` works out exactly, from these rules, what a policy earns per step.
 */
#include "game.hpp"
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <utility>
#include <vector>

namespace games
{
/**
 * @struct access_state
 * @brief what the agent sees: the number of free servers, and the priority of the customer at the head
 */
struct access_state
{
    /// the number of free servers: 0 ... servers
    unsigned int free = 0;
    /// the priority of the customer at the head of the queue: 1, 2, 4 or 8
    unsigned int priority = 1;

    /// @brief two states are equal when both the free servers and the priority are
    bool operator==(const access_state & rhs) const
    {
        return (this->free == rhs.free) &&
               (this->priority == rhs.priority);
    }
};

/**
 * @class access_control
 * @brief accept or reject the customers of a queue, to earn as much as possible per step, forever
 */
class access_control
{
public:
    /// a state is the free servers and the priority of the customer at the head
    using state_type  = access_state;
    /// an action is `reject` or `accept`
    using action_type = unsigned int;
    /// the two actions
    enum : action_type
    {
        reject = 0,
        accept = 1
    };

    /**
     * @brief create the queue, with every server free
     * @param servers is the number of servers (10 in the book)
     * @param free_probability is the probability that a busy server becomes free at each step (0.06 in the book)
     * @param seed seeds the customers' priorities and the servers becoming free
     */
    explicit access_control(unsigned int servers = 10,
                            double free_probability = 0.06,
                            unsigned int seed = 0)
    : servers_(servers),
      free_probability_(free_probability),
      prng_(seed)
    {
        at_.free = servers_;
        at_.priority = next_priority();
    }

    /// @return the number of servers
    unsigned int servers() const
    {
        return servers_;
    }

    /// @return the four priorities, which are also the rewards for accepting: 1, 2, 4 and 8
    static std::vector<unsigned int> priorities()
    {
        return {1, 2, 4, 8};
    }

    /// @return the state the queue is in: a task that never ends does not start again
    state_type reset()
    {
        return at_;
    }

    /// @brief accept or reject the customer at the head @return the state reached and the reward (it never ends)
    step_result<state_type> step(const action_type & a)
    {
        step_result<state_type> out;
        // an accepted customer takes a free server and pays its priority
        if (a == accept && at_.free > 0) {
            at_.free--;
            out.reward = at_.priority;
        }
        // each busy server becomes free with probability p
        std::bernoulli_distribution becomes_free(free_probability_);
        const unsigned int busy = servers_ - at_.free;
        for (unsigned int i = 0; i < busy; i++) {
            if (becomes_free(prng_)) {
                at_.free++;
            }
        }
        // the next customer comes to the head of the queue
        at_.priority = next_priority();
        out.next = at_;
        out.terminated = false;
        return out;
    }

    /// @return the legal actions in @param s: reject, and accept if a server is free
    std::vector<action_type> actions(const state_type & s) const
    {
        if (s.free == 0) {
            return {reject};
        }
        return {reject, accept};
    }

    /**
     * @brief the exact average reward per step of following a policy forever, worked out from the rules
     * @param choose gives the action taken in each state
     * @return ρ, the average reward per step
     *
     * Sampling a policy can be misleading in a task that never ends: a state reached once in
     * a long while can still decide what happens in the long run. So this works it out from the
     * probabilities instead. The next customer's priority does not depend on anything, so the
     * number of free servers alone is a Markov chain. In the long run it has f servers free in a
     * fraction μ(f) of the steps, which solves the balance equations
     *
     * μ(f') = Σ_f μ(f) p(f' | f), with Σ_f μ(f) = 1
     *
     * and the average reward is ρ = Σ_f μ(f) r(f), where r(f) is the mean reward with f servers free.
     */
    double average_reward(const std::function<action_type(const state_type &)> & choose) const
    {
        // if no server ever becomes free, at most `servers` customers are ever served:
        // whatever the policy, it earns nothing in the long run (and the equations below are singular)
        if (free_probability_ <= 0) {
            return 0;
        }
        const std::size_t count = servers_ + 1;
        // p(f' | f) and r(f), averaged over the four priorities of the customer at the head
        std::vector<std::vector<double>> p(count, std::vector<double>(count, 0.0));
        std::vector<double> r(count, 0.0);
        for (unsigned int free = 0; free <= servers_; free++) {
            for (unsigned int priority : priorities()) {
                // an accepted customer takes a server and pays its priority
                const bool taken = free > 0 && choose({free, priority}) == accept;
                const unsigned int left = taken ? free - 1 : free;
                r[free] += taken ? priority / 4.0 : 0.0;
                // then j of the busy servers become free
                const unsigned int busy = servers_ - left;
                for (unsigned int j = 0; j <= busy; j++) {
                    p[free][left + j] += binomial(busy, j) / 4;
                }
            }
        }
        // the balance equations, one row for each f': Σ_f μ(f) p(f' | f) - μ(f') = 0,
        // but the last one is Σ_f μ(f) = 1 instead (the others already imply it)
        std::vector<std::vector<double>> equations(count, std::vector<double>(count + 1, 0.0));
        for (std::size_t to = 0; to < count; to++) {
            for (std::size_t from = 0; from < count; from++) {
                equations[to][from] = p[from][to] - (from == to ? 1.0 : 0.0);
            }
        }
        for (std::size_t from = 0; from < count; from++) {
            equations[count - 1][from] = 1;
        }
        equations[count - 1][count] = 1;
        auto mu = solve(equations);
        // ρ = Σ_f μ(f) r(f)
        double rho = 0;
        for (std::size_t free = 0; free < count; free++) {
            rho += mu[free] * r[free];
        }
        return rho;
    }

private:
    /// @return the probability that @param j of @param busy servers become free in one step
    double binomial(unsigned int busy,
                    unsigned int j) const
    {
        // C(busy, j) p^j (1 - p)^(busy - j)
        double ways = 1;
        for (unsigned int i = 1; i <= j; i++) {
            ways = ways * static_cast<double>(busy - j + i) / static_cast<double>(i);
        }
        return ways * std::pow(free_probability_, j) * std::pow(1 - free_probability_, busy - j);
    }

    /**
     * @brief solve linear equations by Gaussian elimination, with partial pivoting
     * @param equations has one row per equation: its coefficients, then its right-hand side
     * @return the solution
     */
    static std::vector<double> solve(std::vector<std::vector<double>> equations)
    {
        const std::size_t count = equations.size();
        for (std::size_t column = 0; column < count; column++) {
            // swap in the row with the largest coefficient in this column
            std::size_t pivot = column;
            for (std::size_t row = column + 1; row < count; row++) {
                if (std::fabs(equations[row][column]) > std::fabs(equations[pivot][column])) {
                    pivot = row;
                }
            }
            std::swap(equations[column], equations[pivot]);
            // remove this column from the rows below
            for (std::size_t row = column + 1; row < count; row++) {
                const double factor = equations[row][column] / equations[column][column];
                for (std::size_t c = column; c <= count; c++) {
                    equations[row][c] -= factor * equations[column][c];
                }
            }
        }
        // substitute back, from the last row up
        std::vector<double> x(count, 0.0);
        for (std::size_t row = count; row-- > 0; ) {
            double sum = equations[row][count];
            for (std::size_t c = row + 1; c < count; c++) {
                sum -= equations[row][c] * x[c];
            }
            x[row] = sum / equations[row][row];
        }
        return x;
    }

    /// @return the priority of a new customer: 1, 2, 4 or 8, all equally likely
    unsigned int next_priority()
    {
        std::uniform_int_distribution<unsigned int> pick(0, 3);
        return 1u << pick(prng_);
    }

    // the number of servers
    unsigned int servers_;
    // the probability that a busy server becomes free at each step
    double free_probability_;
    // draws the priorities, and the servers becoming free
    std::mt19937 prng_;
    // the state the queue is in
    state_type at_;
};
} // namespace games

/**
 * Hash specialisation in the STD namespace for struct access_state.
 * It is **required** because relearn stores states in unordered maps,
 * which use hashing functions for the classes they store.
 */
namespace std
{
template <> struct hash<games::access_state>
{
    std::size_t operator()(const games::access_state & s) const
    {
        std::size_t seed = 0;
        relearn::hash_combine(seed, s.free);
        relearn::hash_combine(seed, s.priority);
        return seed;
    }
};
}
#endif
