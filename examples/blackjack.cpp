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

    bool operator==(const card & rhs) const
    {
        return this->name == rhs.name &&
               this->label == rhs.label &&
               this->value == rhs.value;

    }
};

//
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
    bool draw()
    {
        // `hand` is publicly inherited
        //  so we can use it to create a new state
        //  and then randomly decide an action (draw/stay)
        //  until we have a best action for a given state
        return false;
    }

    // return a state by casting self to base class
    relearn::state<hand> state() const
    {
        return relearn::state<hand>(*this);
    }
};

//
//
int main(void)
{
    // PRNG needed for, magic random voodoo
    std::mt19937 gen(static_cast<std::size_t>(std::chrono::high_resolution_clock::now()
                                                           .time_since_epoch().count()));
    // create the dealer, and two players...
    auto dealer = std::make_shared<house>(cards, gen);
    auto agent = std::make_shared<client>();

    // alias state, action: 
    // - a state is the current hand
    // - an action is draw(true) or stay(false)
    using state  = relearn::state<hand>;
    using action = relearn::action<bool>;
    using link   = relearn::link<state,action>;
    // policy memory
    relearn::policy<state,action> policies;
    std::deque<std::deque<link>>  experience;
    
    start:
    // play 10 rounds
    for (int i = 0; i < 10; i++) {
        std::deque<link> episode;

        // one card to dealer/house
        dealer->reset_deck();
        dealer->insert(dealer->deal());

        // two cards to player
        agent->insert(dealer->deal());
        agent->insert(dealer->deal());

        // root state is starting hand
        auto s_t = agent->state();

        play:
        // agent decides to draw
        if (agent->draw()) {
            episode.push_back(link{s_t, action(true)});
            agent->insert(dealer->deal());
            s_t = agent->state();
            goto play;
        }
        else {
            episode.push_back(link{s_t, action(false)});
        }

        // dealer's turn
        while (dealer->draw()) {
            dealer->insert(dealer->deal());
        }

        std::cout << "\t\033[1;34m player's hand: ";
        agent->print();
        std::cout << "\033[0m";
        std::cout << "\t\033[1;35m dealer's hand: ";
        dealer->print();
        std::cout << "\033[0m\n";

        if (hand_compare(*agent, *dealer)) {
            std::cout << "\033[1;32m player wins (•̀ᴗ•́)\033\[0m\r\n";
        }
        else {
            std::cout << "\033[1;31m dealer wins (◕︵◕)\033\[0m\r\n";
        }

        // clear current hand for both players
        agent->clear();
        dealer->clear();
        experience.push_back(episode);
    }

    // at this point, we have some playing experience, which we're going to use
    // in order to train the agent.
    relearn::q_probabilistic<state,action> learner;
    for (auto & episode : experience) {
        for (int i = 0; i < 10; i++) {
            learner(episode, policies);
        }
    }

    return 0;
}
