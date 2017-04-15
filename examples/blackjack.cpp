/**
 * A Blackjack/21 example showing how non-deterministic (probabilistic)
 * Episodic Q-Learning works.
 * 
 * @version 0.1.0
 * @author Alex Giokas
 * @date 19.11.2016
 */
#include <iostream>
#include <string>
#include <deque>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>
#include "../src/relearn.hpp"

// a simple card structure
struct card
{
    std::string name;
    std::string label;
    std::vector<unsigned int> value;

    void print() const
    {
        std::cout << name << " "
                  << label << std::endl;
    }             
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
    // calculate value of hand - use max value of hand
    unsigned int max_value() const
    {
        unsigned int result = 0;
        for (const card & k : cards) {
            result += *std::max_element(k.value.begin(), k.value.end()); 
        }
        return result; 
    }

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

private:
    std::vector<card> cards;
};

//
// calculate probability of a card feature (name, label, value)
// by examining the left (unseen) cards
template <typename T>
struct probability
{
    float operator()(const T, const std::vector<card> cards);
};

//
// Base class for all players
//
struct player
{
    virtual bool draw(hand)
    { return false; }

    void start(card dealt)
    {
        my_hand.insert(dealt);
    }

    hand my_hand;
};

//
// House/dealer only uses simple rules to draw or stay
struct house : public player
{
    house(std::deque<card> cards, std::mt19937 & prng)
    : cards(cards), gen(prng)
    {}

    // draw a card based on current hand - house always draws until 17 is reached
    bool draw(hand opponent)
    {
        return (my_hand.min_value() < 17 || my_hand.max_value() < 17);
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
// a probability based player (classical probs)
//
struct classic : public player
{
    // decide on drawing or staying
    bool draw(hand opponent)
    {
        // TODO: play the probabilities
        return false;
    }

    // cards seen in a round
    std::vector<card> seen;
};

//
// a learnign player (Q-learning) adapting and trying to maximize
// its rewards based on the episodes observed
struct adaptive : public player
{
    // decide on drawing on staying
    bool draw(hand opponent)
    {
        // TODO: play and learn to adapt
        return false;
    }

    // TODO: declare intrnally what a state and action is
    //       state is hand's min_value and max_value (not cards)
    //       adding actual cards will increase state complexity for no apparent benefit(?)
    //       actions is "draw" or "stay" - could also be "split"

    // cards seen in a round
    std::vector<card> seen;
    // TODO: we also need a memory policy,
    //       episodes observed
    //       and the q_learning_probablistic
};


//
// TODO: implement Q-Learning using NON-DETERMINISTIC formula
//       infer winner (21, higher card, blackjack, etc)
//
int main(void)
{
    //
    // a 52 playing card constant vector with unicode symbols :-D
    //
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
    
    // Mersenne twister PRNG
    std::mt19937 gen(static_cast<std::size_t>(std::chrono::high_resolution_clock::now()
                                                           .time_since_epoch().count()));
    // create the dealer, and two players...
    auto dealer = house(cards, gen);
    std::vector<std::shared_ptr<player>> players = {std::make_shared<classic>(), std::make_shared<adaptive>()};
    
    //
    // play one round:
    // #1 house deals one card to a player
    // #2 house plays asks player to draw
    // #3 when player stops, house plays - unless player hits blackjack
    //

    return 0;
}
