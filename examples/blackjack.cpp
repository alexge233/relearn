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

struct card
{
    std::string name;
    std::string label;
    std::vector<unsigned int> value;
};

// TODO
struct hand
{
    // hands are equal
    bool operator==(const hand & arg) const;
    // hand is smaller
    bool operator<(const hand & arg) const;
    // print on stdout
    void print() const;
    // add new card
    void insert(card arg);
    // clear hand
    void clear();
    // hand is 21
    bool is_21() const;
    // hand is blackjack
    bool blackjack() const;
private:
    std::vector<card>;
};

//
// calculate probability of a card feature (name, label, value)
// by examining the left (unseen) cards
// TODO
template <typename T>
struct probability
{
    float operator()(const T,
                     const std::vector<card> cards);
};

//
// the base class of a `player` 
//
template <class T>
struct player
{
    bool draw()
    {
        return static_cast<T*>(this)->draw();
    }
};

//
// House only uses simple rules to draw or stay
// TODO
struct house 
: public player<house>
{
    // draw a card based on current hand
    bool draw();
    // deal a card using current deck
    card deal();
    // shuffle cards randomly
    void reset();
private:
    std::deque<card> cards;
};

//
// a probability based player (classical probs)
//
struct classic 
: public player<classic>
{
    // decide on drawing or staying
    bool draw();
    // deck was reset/shuffled
    void reset();
private:
    std::vector<card> seen;
};

//
// a learnign player (Q-learning) adapting and trying to maximize
// its rewards based on the episodes observed
// TODO
struct adaptive 
: public player<adaptive>
{
    // decide on drawing on staying
    bool draw();
    // deck was reset/shuffled
    void reset();
private:
    std::vector<card> seen;
};


// TODO: implement Q-Learning using NON-DETERMINISTIC formula
//       infer winner (21, higher card, blackjack, etc)

int main(void)
{
    //
    // a 52 playing card constant vector with unicode symbols :-D
    //
    const std::vector<card> cards {
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

    std::cout << "cards: " << cards.size() << std::endl;
    return 0;
}
