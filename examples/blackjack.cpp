/**
 * A Blackjack/21 example showing how non-deterministic (probabilistic)
 * Episodic Q-Learning works.
 * 
 * @version 0.1.0
 * @author Alex Giokas
 * @date 19.11.2016
 */
#include <string>
#include <deque>
#include <vector>

struct card
{
    std::string label;
    std::vector<unsigned int> value;
};

// TODO: implement Q-Learning using NON-DETERMINISTIC formula

// TODO: deal cards from the deck
//       decide on an action (stay, split, pull)
//       infer winner (21, higher card, blackjack, etc)
//
//       Two agents: one is the dealer/house
//                   the other is the player
//                   both create the same episode
//                   but each tries to maximise his reward
//                   
int main(void)
{
    std::vector<card> deck{{"Heart", {1, 11}}, {"Spade", {1, 11}}, {"Diamond", {1, 11}}, {"Club", {1, 11}},
                           {"Heart", {2}}, {"Spade", {2}}, {"Diamond", {2}}, {"Club", {2}}, 
                           {"Heart", {3}}, {"Spade", {3}}, {"Diamond", {3}}, {"Club", {3}},
                            // TODO: ...
                           };
    return 0;
}
