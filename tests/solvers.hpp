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
 * Blackjack (Sutton & Barto, Example 5.1) solved exactly by recursion.
 * The infinite deck makes every card independent: P(10) = 4/13, others 1/13.
 */
class blackjack
{
public:
    blackjack()
    {
        for (unsigned int c = 1; c <= 10; c++) {
            dealer_[c] = dealer_from(c == 1 ? 11 : c, c == 1, 1);
        }
    }

    /// expected reward of sticking with @param sum against dealer card @param showing
    double stick(unsigned int sum, unsigned int showing) const
    {
        const auto & d = dealer_[showing];       // final 17..21 at 0..4, bust at 5
        double value = d[5];                     // dealer busts: win
        for (unsigned int f = 17; f <= 21; f++) {
            double p = d[f - 17];
            if (sum > f) value += p;
            else if (sum < f) value -= p;
        }
        return value;
    }

    /// expected reward of sticking on a natural
    double stick_natural(unsigned int showing) const
    {
        double dealer_natural = showing == 1 ? 4.0 / 13.0 : (showing == 10 ? 1.0 / 13.0 : 0.0);
        return 1.0 - dealer_natural;
    }

    /// expected reward of hitting from (@param sum, @param usable) then playing optimally
    double hit(unsigned int sum, bool usable, unsigned int showing)
    {
        double value = 0;
        for (unsigned int c = 1; c <= 10; c++) {
            auto next = add(sum, usable, c);
            value += p(c) * (next.first > 21 ? -1.0 : best(next.first, next.second, showing));
        }
        return value;
    }

    /// optimal value of a (non-natural) decision state
    double best(unsigned int sum, bool usable, unsigned int showing)
    {
        if (sum < 12) {      // the game always draws below 12
            return hit(sum, usable, showing);
        }
        auto key = std::make_tuple(sum, usable, showing);
        auto it = memo_.find(key);
        if (it != memo_.end()) return it->second;
        double v = std::max(stick(sum, showing), hit(sum, usable, showing));
        memo_[key] = v;
        return v;
    }

    static double p(unsigned int card) { return card == 10 ? 4.0 / 13.0 : 1.0 / 13.0; }

    /// add @param card to a hand (@param sum, @param usable): @return the new (sum, usable)
    static std::pair<unsigned int, bool> add(unsigned int sum, bool usable, unsigned int card)
    {
        sum += card;
        if (card == 1 && !usable && sum + 10 <= 21) {
            sum += 10;
            usable = true;
        }
        if (sum > 21 && usable) {
            sum -= 10;
            usable = false;
        }
        return {sum, usable};
    }

private:
    /// distribution of the dealer's final total: 17..21 at 0..4, bust at 5
    std::array<double, 6> dealer_from(unsigned int sum, bool usable, unsigned int cards) const
    {
        std::array<double, 6> d{};
        if (sum > 21) { d[5] = 1; return d; }
        if (sum >= 17 && cards >= 2) { d[sum - 17] = 1; return d; }
        for (unsigned int c = 1; c <= 10; c++) {
            auto next = add(sum, usable, c);
            auto sub = dealer_from(next.first, next.second, cards + 1);
            for (int i = 0; i < 6; i++) d[i] += p(c) * sub[i];
        }
        return d;
    }

    std::array<std::array<double, 6>, 11> dealer_{};
    std::map<std::tuple<unsigned int, bool, unsigned int>, double> memo_;
};
} // namespace solvers
#endif
