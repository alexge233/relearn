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
/**
 * @struct blackjack_state
 * @brief what the player sees: their total, a usable Ace, the dealer's card, and a natural
 */
struct blackjack_state
{
    /// the player's total: 12 ... 21
    unsigned int player_sum = 0;
    /// true if the player holds an Ace currently counted as 11
    bool usable_ace = false;
    /// the dealer's face-up card: 1 (Ace) ... 10
    unsigned int dealer_showing = 0;
    /// true if the player was dealt Ace + 10-card, and has drawn no card since
    bool natural = false;

    /// @brief two states are equal when everything the player sees is equal
    bool operator==(const blackjack_state & rhs) const
    {
        return (this->player_sum == rhs.player_sum) &&
               (this->usable_ace == rhs.usable_ace) &&
               (this->dealer_showing == rhs.dealer_showing) &&
               (this->natural == rhs.natural);
    }
};

/**
 * @struct blackjack_hand
 * @brief a hand's total, counting one Ace as 11 when that does not bust
 */
struct blackjack_hand
{
    /// the total of the hand
    unsigned int sum = 0;
    /// true if an Ace is counted as 11
    bool usable_ace = false;
    /// the number of cards in the hand
    unsigned int cards = 0;

    /// @brief add @param card (1 for an Ace ... 10) to the hand
    void add(unsigned int card)
    {
        cards++;
        sum += card;
        // count a new Ace as 11 if that does not bust, and if no Ace already counts 11
        if (card == 1 && !usable_ace && sum + 10 <= 21) {
            sum += 10;
            usable_ace = true;
        }
        // over 21 with an Ace counted as 11: count it as 1 instead
        if (sum > 21 && usable_ace) {
            sum -= 10;
            usable_ace = false;
        }
    }

    /// @return true if the hand is over 21
    bool bust() const
    {
        return sum > 21;
    }

    /// @return true if the hand is a natural: 21 with its first two cards
    bool natural() const
    {
        return cards == 2 && sum == 21;
    }
};

/**
 * @class blackjack
 * @brief the game of Blackjack: one player against the dealer, with an infinite deck
 */
class blackjack
{
public:
    /// a state is what the player sees
    using state_type  = blackjack_state;
    /// an action is `stick` or `hit`
    using action_type = unsigned int;
    /// the two actions: stop drawing, or draw a card
    enum : action_type
    {
        stick = 0,
        hit   = 1
    };

    /// @brief a new game, its cards drawn with a generator seeded with @param seed
    explicit blackjack(unsigned int seed = 0)
    : prng_(seed)
    {}

    /// @brief deal a new game @return what the player sees
    state_type reset()
    {
        // two cards to the player
        player_ = blackjack_hand();
        dealer_ = blackjack_hand();
        player_.add(draw());
        player_.add(draw());
        natural_ = player_.natural();
        // the dealer's face-up card: an Ace alone counts 11, so it is shown as 1
        dealer_.add(draw());
        showing_ = dealer_.sum == 11 ? 1 : dealer_.sum;
        // the dealer's face-down card
        dealer_.add(draw());
        // below 12 the player cannot bust, so always draws
        while (player_.sum < 12) {
            player_.add(draw());
        }
        return observe();
    }

    /// @brief take action @param a @return what the player sees, the reward and whether the game ended
    step_result<state_type> step(const action_type & a)
    {
        step_result<state_type> out;
        if (a == hit) {
            // hit: draw a card, which loses the game if the player busts
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
        // a natural wins, unless the dealer also has one
        if (natural_) {
            out.reward = dealer_.natural() ? 0 : 1;
            return out;
        }
        // the dealer draws until reaching 17 or more
        while (dealer_.sum < 17) {
            dealer_.add(draw());
        }
        // the higher total wins, and a dealer who busts loses
        if (dealer_.bust() || player_.sum > dealer_.sum) {
            out.reward = 1;
        }
        else if (player_.sum < dealer_.sum) {
            out.reward = -1;
        }
        return out;
    }

    /// @return the legal actions: `stick` and `hit`, in every state
    std::vector<action_type> actions(const state_type &) const
    {
        return {stick, hit};
    }

    /// @return the probability of drawing card value @param card, 1 ... 10 (10 includes Jack, Queen, King)
    static double card_probability(unsigned int card)
    {
        return card == 10 ? 4.0 / 13.0 : 1.0 / 13.0;
    }

private:
    /// @return a card drawn from an infinite deck: 1 (Ace) ... 10
    unsigned int draw()
    {
        // 13 ranks, of which 10, Jack, Queen and King all count 10
        std::uniform_int_distribution<unsigned int> rank(1, 13);
        unsigned int r = rank(prng_);
        return r > 10 ? 10 : r;
    }

    /// @return what the player sees
    state_type observe() const
    {
        state_type s;
        s.player_sum     = player_.sum;
        s.usable_ace     = player_.usable_ace;
        s.dealer_showing = showing_;
        s.natural        = natural_;
        return s;
    }

    // the generator that draws the cards
    std::mt19937 prng_;
    // the player's hand
    blackjack_hand player_;
    // the dealer's hand
    blackjack_hand dealer_;
    // the dealer's face-up card
    unsigned int showing_ = 0;
    // true while the player holds a natural
    bool natural_ = false;
};
} // namespace games

/**
 * Hash specialisation in the STD namespace for struct blackjack_state.
 * It is **required** because relearn stores states in unordered maps,
 * which use hashing functions for the classes they store.
 */
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
