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

//
// a 52 playing card constant vector with unicode symbols :-D
//
const std::vector<card> cards{{"Ace",  "♠", {1, 11}}, {"Ace",  "♥", {1, 11}}, {"Ace",  "♦", {1, 11}}, {"Ace",  "♣", {1, 11}},
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
                              {"King", "♠", {10}},    {"King", "♥", {10}},    {"King", "♦", {10}},    {"King", "♣", {10}}};

//
// TODO: calculate the probability of a card, given the deck and observed cards
// SEE: http://www.math-only-math.com/playing-cards-probability.html
//

// TODO: create a "House" agent which uses heuristics

// TODO: implement Q-Learning using NON-DETERMINISTIC formula

// TODO: deal cards from the deck
//       decide on an action (stay, pull)
//       infer winner (21, higher card, blackjack, etc)
//
//       Two agents: one is the dealer/house
//                   the other is the player
//                   both create the same episode
//                   but each tries to maximise his reward
//                   
int main(void)
{
    std::cout << "cards: " << cards.size() << std::endl;
    return 0;
}
