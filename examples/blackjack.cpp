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
                  << label << " ";
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

private:
    std::vector<card> cards;
};

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

//
// Base class for all players
//
struct player : public hand
{
    virtual bool draw()
    {
        return false;
    }
};

//
// House/dealer only uses simple rules to draw or stay
//
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
// a probability based player (classical probs)
//
struct classic : public player
{
    // decide on drawing or staying
    bool draw()
    {
        if (!blackjack() && 
            min_value() != 21 &&
            max_value() != 21)
        {
            // find diff of 21 (max) to current hand
            unsigned int min_diff = 21 - min_value();
            unsigned int max_diff = 21 - max_value();
            // TODO:
            // calculate the probability of drawing a 21 or less
            // if that probability is small (less than 50%) don't draw
            // calculate the probability of not getting burnt (drawing a small value)
            // and combine them altogether
        }
        return false;
    }

    // return a deck of cards without the already seen cards
    std::deque<card> remaining_cards()
    {
        std::deque<card> remaining;
        for (const card & k : cards) {
            if (std::find_if(seen.begin(), seen.end(), 
                             [&](const auto obj){
                                 return card_compare(k, obj);    
                             }) == seen.end()) 
            {
                remaining.push_back(k);
            }
        }
        return remaining;
    }

    // calculate Pr of drawing a card with target value, given the cards left
    float probability(unsigned int target, 
                      std::deque<card> left)
    {
        float total = left.size();
        float match = 0;
        for (card k : left) {
             for (auto v : k.value) {
                if (v == target) {
                    match++;
                }
             }
        } 
        return match / total;
    }

    // immutable card set
    const std::deque<card> cards;
    // cards seen in a round
    std::deque<card> seen;
};

//
// a learnign player (Q-learning) adapting and trying to maximize
// its rewards based on the episodes observed
struct adaptive : public player
{
    // decide on drawing on staying
    bool draw()
    {
        // TODO: play and learn to adapt
        return false;
    }

    // TODO: declare intrnally what a state and action is
    //       state is hand's min_value and max_value (not cards)
    //       adding actual cards will increase state complexity for no apparent benefit(?)
    //       actions is "draw" or "stay" - could also be "split"

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
    auto dealer = std::make_shared<house>(cards, gen);
    auto agent = std::make_shared<classic>();
    
    // start playing
    for (int i = 0; i < 10; i++) {
        std::cout << "new game" << std::endl;

        // deal one to self
        dealer->reset_deck();
        auto kard_1 = dealer->deal();
        dealer->insert(kard_1);

        // deal two to player
        auto kard_2 = dealer->deal();
        auto kard_3 = dealer->deal();
        agent->insert(kard_2);
        agent->insert(kard_3);
        agent->seen = {kard_1, kard_2, kard_3};

        // agent print, and decide to draw
        while (agent->draw()) {
            auto kard = dealer->deal();
            agent->insert(kard);
            agent->seen.push_back(kard);
        }
        while (dealer->draw()) {
            dealer->insert(dealer->deal());
        }

        std::cout << "player's hand: ";
        agent->print();
        std::cout << "dealer's hand: ";
        dealer->print();

        if (hand_compare(*agent, *dealer)) {
            std::cout << "player wins!!!" << std::endl;
        }
        else {
            std::cout << "dealer wins :-(" << std::endl;
        }

        agent->clear();
        dealer->clear();
    }

    return 0;
}
