/**
 * A Blackjack/21 example showing how non-deterministic (probabilistic)
 * Episodic Q-Learning works.
 * 
 * @version 0.1.0
 * @author Alex Giokas
 * @date 19.11.2016
 */

struct card
{
    std::string label;
    unsigned int[] value;
};

struct deck
{
    // TODO: setup cards
    // TODO: shuffle them randomly
};

// TODO: deal cards from the deck
//       decide on an action (stay, split, pull)
//       infer winner (21, higher card, blackjack, etc)
//
//       Two agents: one is the dealer/house
//                   the other is the player
//                   both create the same episode
//                   but each tries to maximise his reward
//                   
