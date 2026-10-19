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
};

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
