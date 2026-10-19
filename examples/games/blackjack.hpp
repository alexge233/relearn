#ifndef RELEARN_GAMES_BLACKJACK_HPP
#define RELEARN_GAMES_BLACKJACK_HPP
/**
 * Blackjack, as in Sutton & Barto, Example 5.1.
 *
 * - Infinite deck: cards are drawn with replacement, 10/Jack/Queen/King count 10,
 *   an Ace counts 1 or 11 (it is "usable" while counting it as 11 does not bust).
 * - The player sees their own total, whether they hold a usable Ace, and the
 *   dealer's face-up card. Below 12 the player always draws (it cannot bust), so
 *   decisions start at 12.
 * - A natural (Ace + 10-card on the deal) wins unless the dealer also has one.
 * - The dealer draws until reaching 17 or more, then sticks.
 * - Rewards: +1 win, 0 draw, -1 loss, received when the game ends.
 */
#include "game.hpp"
#include <functional>
#include <random>
#include <vector>

namespace games
{
/// what the player sees
struct blackjack_state
{
    unsigned int player_sum = 0;      // 12 ... 21
    bool usable_ace = false;          // an Ace currently counted as 11
    unsigned int dealer_showing = 0;  // 1 (Ace) ... 10
    bool natural = false;             // Ace + 10-card on the deal, no card drawn since

    bool operator==(const blackjack_state & rhs) const
    {
        return player_sum == rhs.player_sum && usable_ace == rhs.usable_ace &&
               dealer_showing == rhs.dealer_showing && natural == rhs.natural;
    }
};

/// a hand's total, counting one Ace as 11 when that does not bust
struct blackjack_hand
{
    unsigned int sum = 0;
    bool usable_ace = false;
    unsigned int cards = 0;

    void add(unsigned int card)
    {
        cards++;
        sum += card;
        if (card == 1 && !usable_ace && sum + 10 <= 21) {
            sum += 10;
            usable_ace = true;
        }
        if (sum > 21 && usable_ace) {
            sum -= 10;
            usable_ace = false;
        }
    }
    bool bust() const { return sum > 21; }
    bool natural() const { return cards == 2 && sum == 21; }
};

class blackjack
{
public:
    using state_type  = blackjack_state;
    using action_type = unsigned int;
    enum : action_type { stick = 0, hit = 1 };

    explicit blackjack(unsigned int seed = 0)
    : prng_(seed)
    {}

    /// deal a new game
    state_type reset()
    {
        player_ = blackjack_hand();
        dealer_ = blackjack_hand();
        player_.add(draw());
        player_.add(draw());
        natural_ = player_.natural();
        dealer_.add(draw());
        showing_ = dealer_.sum == 11 ? 1 : dealer_.sum;  // an Ace alone counts 11
        dealer_.add(draw());                              // face-down card
        while (player_.sum < 12) {
            player_.add(draw());
        }
        return observe();
    }

    step_result<state_type> step(const action_type & a)
    {
        step_result<state_type> out;
        if (a == hit) {
            natural_ = false;
            player_.add(draw());
            out.next = observe();
            if (player_.bust()) {
                out.reward = -1;
                out.terminated = true;
            }
            return out;
        }
        // stick: the game ends
        out.next = observe();
        out.terminated = true;
        if (natural_) {
            out.reward = dealer_.natural() ? 0 : 1;
            return out;
        }
        while (dealer_.sum < 17) {
            dealer_.add(draw());
        }
        if (dealer_.bust() || player_.sum > dealer_.sum) {
            out.reward = 1;
        }
        else if (player_.sum < dealer_.sum) {
            out.reward = -1;
        }
        return out;
    }

    std::vector<action_type> actions(const state_type &) const
    {
        return {stick, hit};
    }

    /// probability of each card value 1 ... 10 (10 includes Jack, Queen, King)
    static double card_probability(unsigned int card)
    {
        return card == 10 ? 4.0 / 13.0 : 1.0 / 13.0;
    }

private:
    unsigned int draw()
    {
        std::uniform_int_distribution<unsigned int> rank(1, 13);
        unsigned int r = rank(prng_);
        return r > 10 ? 10 : r;
    }

    state_type observe() const
    {
        state_type s;
        s.player_sum = player_.sum;
        s.usable_ace = player_.usable_ace;
        s.dealer_showing = showing_;
        s.natural = natural_;
        return s;
    }

    std::mt19937 prng_;
    blackjack_hand player_;
    blackjack_hand dealer_;
    unsigned int showing_ = 0;
    bool natural_ = false;
};
} // namespace games

namespace std
{
template <> struct hash<games::blackjack_state>
{
    std::size_t operator()(const games::blackjack_state & s) const
    {
        std::size_t seed = 0;
        relearn::hash_combine(seed, s.player_sum);
        relearn::hash_combine(seed, s.usable_ace);
        relearn::hash_combine(seed, s.dealer_showing);
        relearn::hash_combine(seed, s.natural);
        return seed;
    }
};
}
#endif
