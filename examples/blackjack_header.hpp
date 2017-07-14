#ifndef BLACKJACK_HPP
#define BLACKJACK_HPP
#include <iostream>
#include <string>
#include <deque>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include "../src/relearn.hpp"
#if USING_BOOST_SERIALIZATION
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/access.hpp>
#endif
/**
 * The basic Blackjack header structures:
 *  - card
 *  - hand
 *  - player
 *  - house
 *  - client
 *
 *  All that is minimally needed in order to
 *  create the Blackjack example
 */

// a simple card structure
struct card
{
    std::string name;
    std::string label;
    std::vector<unsigned int> value;

    void print() const
    {
        std::cout << name << " "
                  << label << " ";
    }             

    bool operator==(const card & rhs) const
    {
        return this->name == rhs.name &&
               this->label == rhs.label &&
               this->value == rhs.value;

    }
#if USING_BOOST_SERIALIZATION
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & name;
        ar & label;
        ar & value;
    }
#endif
};

// a 52 playing card constant vector with unicode symbols :-D
const std::deque<card> cards {
    {"Ace",  "♠", {1, 11}}, {"Ace",  "♥", {1, 11}}, {"Ace",  "♦", {1, 11}}, {"Ace",  "♣", {1, 11}},
    {"Two",  "♠", {2}},     {"Two",  "♥", {2}},     {"Two",  "♦", {2}},     {"Two",  "♣", {2}}, 
    {"Three","♠", {3}},     {"Three","♥", {3}},     {"Three","♦", {3}},     {"Three","♣", {3}},
    {"Four", "♠", {4}},     {"Four", "♥", {4}},     {"Four", "♦", {4}},     {"Four", "♣", {4}},
    {"Five", "♠", {5}},     {"Five", "♥", {5}},     {"Five", "♦", {5}},     {"Five", "♣", {5}},
    {"Six",  "♠", {6}},     {"Six",  "♥", {6}},     {"Six",  "♦", {6}},     {"Six",  "♣", {6}},
    {"Seven","♠", {7}},     {"Seven","♥", {7}},     {"Seven","♦", {7}},     {"Seven","♣", {7}},
    {"Eight","♠", {8}},     {"Eight","♥", {8}},     {"Eight","♦", {8}},     {"Eight","♣", {8}},
    {"Nine", "♠", {9}},     {"Nine", "♥", {9}},     {"Nine", "♦", {9}},     {"Nine", "♣", {9}},
    {"Ten",  "♠", {10}},    {"Ten",  "♥", {10}},    {"Ten",  "♦", {10}},    {"Ten",  "♣", {10}},
    {"Jack", "♠", {10}},    {"Jack", "♥", {10}},    {"Jack", "♦", {10}},    {"Jack", "♣", {10}},
    {"Queen","♠", {10}},    {"Queen","♥", {10}},    {"Queen","♦", {10}},    {"Queen","♣", {10}},
    {"King", "♠", {10}},    {"King", "♥", {10}},    {"King", "♦", {10}},    {"King", "♣", {10}}
};

bool card_compare(const card & lhs, const card & rhs)
{
    return lhs.name == rhs.name &&
           lhs.label == rhs.label &&
           lhs.value == rhs.value;
}

// hand is the currently held cards
struct hand
{
    hand() = default;
    hand(const hand &) = default;

    // calculate value of hand - use max value of hand
    unsigned int max_value() const
    {
        unsigned int result = 0;
        for (const card & k : cards) {
            result += *std::max_element(k.value.begin(), k.value.end()); 
        }
        return result; 
    }

    // calculate value of hand - use min value (e.g., when hold an Ace)
    unsigned int min_value() const
    {
        unsigned int result = 0;
        for (const card & k : cards) {
            result += *std::min_element(k.value.begin(), k.value.end()); 
        }
        return result; 
    }

    // print on stdout
    void print() const
    {
        for (card k : cards) {
            k.print(); 
        }
        std::cout << std::endl;
    }

    // add new card
    void insert(card arg)
    {
        cards.push_back(arg);
    }

    // clear hand
    void clear()
    {
        cards.clear();
    }

    // hand is blackjack
    bool blackjack() const
    {
        std::vector<card> twoblacks = {{"Ace",  "♠", {1, 11}}, 
                                       {"Ace",  "♣", {1, 11}}};
        return std::is_permutation(twoblacks.begin(), twoblacks.end(), 
                                   cards.begin(),  card_compare);
    }

    // hash this hand for relearn
    std::size_t hash() const
    {
        std::size_t seed = 0;
        for (auto & k : cards) {
            for (auto & v : k.value) {
                relearn::hash_combine(seed, v);
            }
        }
        return seed;
    }

    bool operator==(const hand & rhs) const
    {
        return this->cards == rhs.cards;
    }

private:
    std::vector<card> cards;
#if USING_BOOST_SERIALIZATION
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & cards;
    }
#endif
};

namespace std 
{
template <> struct hash<hand>
{
    std::size_t operator()(hand const& arg) const 
    {
        return arg.hash();
    }
};
}

/// compare hands (return true for lhs wins)
bool hand_compare(const hand & lhs, const hand & rhs)
{
    if (lhs.blackjack()) return true;
    else if (rhs.blackjack()) return false;

    if (lhs.min_value() > 21) return false;
    else if (rhs.min_value() > 21 && lhs.min_value() < 21) return true;

    if (lhs.max_value() > rhs.max_value()) return true;
    else return false;
}

// Base class for all players
struct player : public hand
{
    virtual bool draw()
    {
        return false;
    }
};

// House/dealer only uses simple rules to draw or stay
struct house : public player
{
    house(std::deque<card> cards, std::mt19937 & prng)
    : cards(cards), gen(prng)
    {}

    // draw a card based on current hand - house always draws until 17 is reached
    bool draw()
    {
        return (min_value() < 17 || max_value() < 17);
    }

    // deal a card using current deck - or reset and deal
    card deal()
    {
        if (deck.size() > 0) {
            auto obj = deck.front();
            deck.pop_front();
            return obj;
        }
        else {
            reset_deck();
            return deal();
        }
    }

    // shuffle cards randomly
    void reset_deck()
    {
        deck = cards;
        std::shuffle(std::begin(deck), std::end(deck), gen);
    }

private:
    std::deque<card> deck;
    const std::deque<card> cards;
    std::mt19937 gen;
};

//
// our learning adaptive player
struct client : public player
{
    // decide on drawing or staying
    bool draw(std::mt19937 & prng,
              relearn::state<hand> s_t,
              relearn::policy<relearn::state<hand>,
                              relearn::action<bool>> & map)
    {
        auto pair = map.best(s_t);
        std::uniform_real_distribution<float> dist(0, 1);
        // there exists a "best action" and it is positive
        if (pair.first && pair.second > 0) {
            policy_actions++;
            return pair.first->trait(); 
        }
        // there does not exist a "best action"
        else {
            random_actions++;
            return (dist(prng) > 0.5 ? true : false);
        }
    }

    // return a state by casting self to base class
    relearn::state<hand> state() const
    {
        return relearn::state<hand>(*this);
    }

    void reset()
    {
        random_actions = 0;
        policy_actions = 0;
    }

    float random_actions = 0;
    float policy_actions = 0;
};

#endif
