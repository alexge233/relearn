#ifndef RELEARN_TESTS_SOLVERS_HPP
#define RELEARN_TESTS_SOLVERS_HPP
/**
 * Exact solutions of the example games, used as ground truth by the tests.
 * They are written from each game's rules, independently of the game classes
 * and of relearn's learners.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace solvers
{
/**
 * @struct outcome
 * @brief one possible outcome of taking an action in a tabular model
 */
struct outcome
{
    /// probability of this outcome
    double probability;
    /// index of the next state
    int next;
    /// reward received
    double reward;
    /// the episode ends: the next state is worth 0
    bool terminal;
};

/**
 * @struct tabular
 * @brief a finite MDP with states 0 ... states-1 and actions 0 ... actions-1
 *
 * `model(s, a)` lists the outcomes of action a in state s; an empty list
 * means the action is not allowed there (or s is terminal).
 */
struct tabular
{
    int states = 0;
    int actions = 0;
    double gamma = 1;
    std::function<std::vector<outcome>(int, int)> model;

    /**
     * @brief the value of taking an action, given the values of the states:
     *        Q(s,a) = Σ p(s',r | s,a) [r + γ V(s')]   (Sutton & Barto, Section 3.5)
     * @param v are the state values
     * @param s is the state
     * @param a is the action
     * @return the value, or NaN if @param a is not allowed in @param s
     */
    double q(const std::vector<double> & v,
             int s,
             int a) const
    {
        auto outcomes = model(s, a);
        if (outcomes.empty()) {
            return std::nan("");
        }
        double value = 0;
        for (const auto & o : outcomes) {
            // a terminal next state is worth 0: the episode ends there
            value += o.probability * (o.reward + (o.terminal ? 0.0 : gamma * v[o.next]));
        }
        return value;
    }

    /**
     * @brief the optimal state values, by value iteration (Sutton & Barto, Section 4.4)
     * @param tolerance stops the sweeps once no value changes by more than this
     * @param max_sweeps is the most sweeps over the states
     * @return the value of every state
     */
    std::vector<double> optimal_values(double tolerance = 1e-12,
                                       int max_sweeps = 100000) const
    {
        std::vector<double> v(states, 0.0);
        for (int sweep = 0; sweep < max_sweeps; sweep++) {
            double change = 0;
            for (int s = 0; s < states; s++) {
                // V(s) = max over the allowed actions of Q(s,a)
                double best = -1e300;
                bool any = false;
                for (int a = 0; a < actions; a++) {
                    double value = q(v, s, a);
                    if (!std::isnan(value)) {
                        best = std::max(best, value);
                        any = true;
                    }
                }
                // a terminal state (no allowed action) keeps its value of 0
                if (!any) {
                    continue;
                }
                change = std::max(change, std::fabs(best - v[s]));
                v[s] = best;
            }
            if (change < tolerance) {
                break;
            }
        }
        return v;
    }

    /**
     * @brief the state values of following a policy, by iterative evaluation (Sutton & Barto, Section 4.1)
     * @param policy is one action per state
     * @param tolerance stops the sweeps once no value changes by more than this
     * @param max_sweeps is the most sweeps over the states
     * @param floor keeps the values finite: a policy that never ends an episode
     *        (with gamma = 1) keeps falling below it
     * @return the value of every state
     */
    std::vector<double> policy_values(const std::vector<int> & policy,
                                      double tolerance = 1e-12,
                                      int max_sweeps = 100000,
                                      double floor = -1e6) const
    {
        std::vector<double> v(states, 0.0);
        for (int sweep = 0; sweep < max_sweeps; sweep++) {
            double change = 0;
            for (int s = 0; s < states; s++) {
                // V(s) = Q(s, policy(s))
                double value = q(v, s, policy[s]);
                if (std::isnan(value)) {
                    continue;
                }
                value = std::max(value, floor);
                change = std::max(change, std::fabs(value - v[s]));
                v[s] = value;
            }
            if (change < tolerance) {
                break;
            }
        }
        return v;
    }

    /**
     * @brief the state values of following a stochastic policy, by iterative evaluation
     *        (Sutton & Barto, Section 4.1): V(s) = Σ_a π(a | s) Q(s,a)
     * @param pi gives, for every state, the probability of each action
     * @param tolerance stops the sweeps once no value changes by more than this
     * @param max_sweeps is the most sweeps over the states
     * @return the value of every state
     */
    std::vector<double> policy_values(const std::vector<std::vector<double>> & pi,
                                      double tolerance = 1e-12,
                                      int max_sweeps = 100000) const
    {
        std::vector<double> v(states, 0.0);
        for (int sweep = 0; sweep < max_sweeps; sweep++) {
            double change = 0;
            for (int s = 0; s < states; s++) {
                // V(s) = Σ_a π(a | s) Q(s,a), over the allowed actions
                double value = 0;
                bool any = false;
                for (int a = 0; a < actions; a++) {
                    double q_value = q(v, s, a);
                    if (!std::isnan(q_value)) {
                        value += pi[s][a] * q_value;
                        any = true;
                    }
                }
                // a terminal state (no allowed action) keeps its value of 0
                if (!any) {
                    continue;
                }
                change = std::max(change, std::fabs(value - v[s]));
                v[s] = value;
            }
            if (change < tolerance) {
                break;
            }
        }
        return v;
    }

    /**
     * @struct average
     * @brief the optimal average reward per step of a task that never ends, its differential values and its policy
     */
    struct average
    {
        /// ρ*, the optimal average reward per step: halfway between `low` and `high`
        double rho = 0;
        /// a lower bound of ρ*: the smallest (T h)(s) - h(s) of the last sweep
        double low = 0;
        /// an upper bound of ρ*: the largest (T h)(s) - h(s) of the last sweep
        double high = 0;
        /// h(s), the differential value of every state, relative to state 0 (h(0) = 0)
        std::vector<double> h;
        /// the best action of every state, given h
        std::vector<int> policy;
        /// the sweeps done
        int sweeps = 0;
    };

    /**
     * @brief the optimal average reward of a task that never ends, by relative value iteration
     *        (Puterman, 1994, Chapter 8): it needs `gamma` = 1 and no terminal outcome
     * @param tolerance stops the sweeps once the bounds of ρ* are closer than this
     * @param max_sweeps is the most sweeps over the states
     * @return the optimal ρ* (with its bounds), the differential values h, and an optimal policy
     *
     * Each sweep works out, for every state, (T h)(s) = max_a Σ p(s',r | s,a) [r + h(s')],
     * and then h(s) = (T h)(s) - (T h)(0). The smallest and the largest (T h)(s) - h(s) are
     * a lower and an upper bound of ρ*; when every policy reaches one same set of states
     * (unichain), and does not cycle (aperiodic), they close in on it. Do check `low` and `high`:
     * if the sweeps run out first, ρ* is somewhere between them.
     */
    average optimal_average(double tolerance = 1e-12,
                            int max_sweeps = 100000) const
    {
        average result;
        const std::size_t count = static_cast<std::size_t>(states);
        result.h.assign(count, 0.0);
        result.policy.assign(count, -1);
        std::vector<double> next(count, 0.0);
        for (int sweep = 1; sweep <= max_sweeps; sweep++) {
            result.sweeps = sweep;
            for (std::size_t s = 0; s < count; s++) {
                // (T h)(s) = max over the allowed actions of Σ p [r + h(s')]
                double best = -1e300;
                for (int a = 0; a < actions; a++) {
                    double value = q(result.h, static_cast<int>(s), a);
                    if (!std::isnan(value) && value > best) {
                        best = value;
                        result.policy[s] = a;
                    }
                }
                next[s] = best;
            }
            // T h - h brackets ρ*: its smallest and largest values are the bounds
            result.low  = 1e300;
            result.high = -1e300;
            for (std::size_t s = 0; s < count; s++) {
                result.low  = std::min(result.low, next[s] - result.h[s]);
                result.high = std::max(result.high, next[s] - result.h[s]);
            }
            // keep h relative to state 0, so that it stays finite
            for (std::size_t s = 0; s < count; s++) {
                result.h[s] = next[s] - next[0];
            }
            if (result.high - result.low < tolerance) {
                break;
            }
        }
        result.rho = (result.low + result.high) / 2;
        return result;
    }

    /**
     * @brief how often a policy visits each state in the long run, in a task that never ends
     * @param policy is one action per state
     * @return μ(s), the fraction of the steps spent in each state: the solution of
     *         μ(s') = Σ_s μ(s) p(s' | s, policy(s)) with Σ_s μ(s) = 1, by Gaussian elimination
     *
     * It is unique when the policy reaches one same set of states from every state (unichain).
     * Unlike iterating, solving the equations does not slow down when a state is reached only
     * very rarely, e.g. a state the policy never leaves, which it reaches once in a long while.
     */
    std::vector<double> stationary(const std::vector<int> & policy) const
    {
        const std::size_t count = static_cast<std::size_t>(states);
        // the equations, one row each: Σ_s μ(s) p(s' | s) - μ(s') = 0 for every s' but the last,
        // and Σ_s μ(s) = 1 instead of the last one (which the others already imply)
        std::vector<std::vector<double>> equations(count, std::vector<double>(count + 1, 0.0));
        for (std::size_t s = 0; s < count; s++) {
            equations[s][s] -= 1;
            for (const auto & o : model(static_cast<int>(s), policy[s])) {
                equations[static_cast<std::size_t>(o.next)][s] += o.probability;
            }
        }
        for (std::size_t s = 0; s < count; s++) {
            equations[count - 1][s] = 1;
        }
        equations[count - 1][count] = 1;
        // Gaussian elimination, with partial pivoting
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
        std::vector<double> mu(count, 0.0);
        for (std::size_t row = count; row-- > 0; ) {
            double sum = equations[row][count];
            for (std::size_t c = row + 1; c < count; c++) {
                sum -= equations[row][c] * mu[c];
            }
            mu[row] = sum / equations[row][row];
        }
        return mu;
    }

    /**
     * @brief the average reward per step of following a policy forever, in a task that never ends
     * @param policy is one action per state
     * @return ρ = Σ_s μ(s) Σ p(s',r | s, policy(s)) r, with μ the fraction of the steps spent in each state
     */
    double policy_average(const std::vector<int> & policy) const
    {
        auto mu = stationary(policy);
        double rho = 0;
        for (std::size_t s = 0; s < mu.size(); s++) {
            for (const auto & o : model(static_cast<int>(s), policy[s])) {
                rho += mu[s] * o.probability * o.reward;
            }
        }
        return rho;
    }
};

/**
 * @brief the access-control queuing task (Sutton & Barto, 1st edition Example 6.7,
 *        2nd edition Example 10.2), as a tabular model, written from its rules
 * @param servers is the number of servers
 * @param p is the probability that a busy server becomes free at each step
 * @return the model: state = 4 * free servers + k, for the priority 2^k of the customer
 *         at the head; action 0 rejects, 1 accepts (not allowed with no free server)
 *
 * Accepting takes a free server and pays the priority; then each of the b busy servers
 * (including one just taken) becomes free with probability p, so that k of them do with the
 * binomial probability C(b,k) p^k (1-p)^(b-k), and the next priority is 1, 2, 4 or 8 with
 * probability 1/4 each.
 */
inline tabular access_control(int servers = 10,
                              double p = 0.06)
{
    tabular mdp;
    mdp.states  = 4 * (servers + 1);
    mdp.actions = 2;
    mdp.gamma   = 1;
    mdp.model = [servers, p](int s, int a) {
        std::vector<outcome> outcomes;
        const int free = s / 4;
        const int priority = 1 << (s % 4);
        // with no free server, a customer cannot be accepted
        if (a == 1 && free == 0) {
            return outcomes;
        }
        // accepting takes a server and pays the priority
        const int taken  = a == 1 ? 1 : 0;
        const double reward = a == 1 ? priority : 0;
        const int busy = servers - (free - taken);
        // k of the busy servers become free, and the next customer has priority 2^j
        for (int k = 0; k <= busy; k++) {
            // C(busy, k) p^k (1 - p)^(busy - k)
            double choose = 1;
            for (int i = 1; i <= k; i++) {
                choose = choose * (busy - k + i) / i;
            }
            const double p_k = choose * std::pow(p, k) * std::pow(1 - p, busy - k);
            for (int j = 0; j < 4; j++) {
                outcomes.push_back({p_k / 4, 4 * (free - taken + k) + j, reward, false});
            }
        }
        return outcomes;
    };
    return mdp;
}

/**
 * @class blackjack
 * @brief Blackjack (Sutton & Barto, Example 5.1) solved exactly by recursion
 *
 * The infinite deck makes every card independent: P(10) = 4/13, others 1/13.
 */
class blackjack
{
public:
    /// @brief work out the dealer's final totals for every card the dealer can show
    blackjack()
    {
        for (unsigned int c = 1; c <= 10; c++) {
            // an Ace starts the dealer's hand at 11, with a usable Ace
            dealer_[c] = dealer_from(c == 1 ? 11 : c, c == 1, 1);
        }
    }

    /**
     * @brief the expected reward of sticking
     * @param sum is the player's total
     * @param showing is the dealer's card
     * @return the expected reward
     */
    double stick(unsigned int sum,
                 unsigned int showing) const
    {
        // final 17..21 at 0..4, bust at 5
        const auto & d = dealer_[showing];
        // dealer busts: win
        double value = d[5];
        for (unsigned int f = 17; f <= 21; f++) {
            double p = d[f - 17];
            if (sum > f) {
                value += p;
            }
            else if (sum < f) {
                value -= p;
            }
        }
        return value;
    }

    /// @return the expected reward of sticking on a natural, against dealer card @param showing
    double stick_natural(unsigned int showing) const
    {
        // a natural only draws against a dealer's natural: an Ace with a 10, or a 10 with an Ace
        double dealer_natural = showing == 1 ? 4.0 / 13.0 : (showing == 10 ? 1.0 / 13.0 : 0.0);
        return 1.0 - dealer_natural;
    }

    /**
     * @brief the expected reward of hitting, then playing optimally
     * @param sum is the player's total
     * @param usable is true if the player has a usable Ace
     * @param showing is the dealer's card
     * @return the expected reward
     */
    double hit(unsigned int sum,
               bool usable,
               unsigned int showing)
    {
        double value = 0;
        for (unsigned int c = 1; c <= 10; c++) {
            // draw card c: bust loses, otherwise play on optimally
            auto next = add(sum, usable, c);
            value += p(c) * (next.first > 21 ? -1.0 : best(next.first, next.second, showing));
        }
        return value;
    }

    /**
     * @brief the optimal value of a (non-natural) decision state
     * @param sum is the player's total
     * @param usable is true if the player has a usable Ace
     * @param showing is the dealer's card
     * @return the larger of the values of sticking and hitting
     */
    double best(unsigned int sum,
                bool usable,
                unsigned int showing)
    {
        // the game always draws below 12
        if (sum < 12) {
            return hit(sum, usable, showing);
        }
        // each state is worked out once, then remembered
        auto key = std::make_tuple(sum, usable, showing);
        auto it = memo_.find(key);
        if (it != memo_.end()) {
            return it->second;
        }
        double v = std::max(stick(sum, showing), hit(sum, usable, showing));
        memo_[key] = v;
        return v;
    }

    /// @return the probability of drawing @param card (1 is an Ace, 10 any ten-valued card)
    static double p(unsigned int card)
    {
        return card == 10 ? 4.0 / 13.0 : 1.0 / 13.0;
    }

    /**
     * @brief add a card to a hand
     * @param sum is the hand's total
     * @param usable is true if the hand has an Ace counted as 11
     * @param card is the card drawn
     * @return the new (sum, usable)
     */
    static std::pair<unsigned int, bool> add(unsigned int sum,
                                             bool usable,
                                             unsigned int card)
    {
        sum += card;
        // an Ace counts 11 when that does not bust the hand
        if (card == 1 && !usable && sum + 10 <= 21) {
            sum += 10;
            usable = true;
        }
        // about to bust: a usable Ace drops back to 1
        if (sum > 21 && usable) {
            sum -= 10;
            usable = false;
        }
        return {sum, usable};
    }

private:
    /**
     * @brief the distribution of the dealer's final total
     * @param sum is the dealer's total so far
     * @param usable is true if the dealer has a usable Ace
     * @param cards is how many cards the dealer holds
     * @return the probabilities of final totals 17..21 at 0..4, and of a bust at 5
     */
    std::array<double, 6> dealer_from(unsigned int sum,
                                      bool usable,
                                      unsigned int cards) const
    {
        std::array<double, 6> d{};
        // the dealer busts
        if (sum > 21) {
            d[5] = 1;
            return d;
        }
        // the dealer sticks on 17 or more (once holding two cards)
        if (sum >= 17 && cards >= 2) {
            d[sum - 17] = 1;
            return d;
        }
        // otherwise the dealer draws: average over every card
        for (unsigned int c = 1; c <= 10; c++) {
            auto next = add(sum, usable, c);
            auto sub = dealer_from(next.first, next.second, cards + 1);
            for (int i = 0; i < 6; i++) {
                d[i] += p(c) * sub[i];
            }
        }
        return d;
    }

    // the dealer's final totals, for each card the dealer shows
    std::array<std::array<double, 6>, 11> dealer_{};
    // the optimal values already worked out
    std::map<std::tuple<unsigned int, bool, unsigned int>, double> memo_;
};
} // namespace solvers
#endif
