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
        auto a_t = map.best_action(s_t);
        auto q_v = map.best_value(s_t);
        std::uniform_real_distribution<float> dist(0, 1);
        // there exists a "best action" and it is positive
        if (a_t && q_v > 0) {
            sum_q += q_v;
            policy_actions++;
            return a_t->trait(); 
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

    float random_actions = 0;
    float policy_actions = 0;
    float sum_q = 0;
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
    
    float sum  = 0;
    float wins = 0;
    std::cout << "starting! Press CTRL-C to stop at any time!" 
              << std::endl;
    start:
    // play 10 rounds - then stop
    for (int i = 0; i < 10; i++) {
        sum++;
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
        // if agent's hand is burnt skip all else
        if (agent->min_value() && agent->max_value() > 21) {
            goto cmp;
        }
        // agent decides to draw        
        if (agent->draw(gen, s_t, policies)) {
            episode.push_back(link{s_t, action(true)});
            agent->insert(dealer->deal());
            s_t = agent->state();
            goto play;
        }
        // agent decides to stay
        else {
            episode.push_back(link{s_t, action(false)});
        }
        // dealer's turn
        while (dealer->draw()) {
            dealer->insert(dealer->deal());
        }

        cmp:
        // compare hands, assign rewards!
        if (hand_compare(*agent, *dealer)) {
            if (!episode.empty()) {
                episode.back().state.set_reward(1); 
            }
            wins++;
        }
        else {
            if (!episode.empty()) {
                episode.back().state.set_reward(-1); 
            }
        }

        // clear current hand for both players
        agent->clear();
        dealer->clear();
        experience.push_back(episode);
        std::cout << "\twin ratio: " << wins / sum << std::endl;
        std::cout << "\ton-policy ratio: " 
                  << agent->policy_actions / (agent->policy_actions + agent->random_actions) 
                  << std::endl;
        std::cout << "\tavg Q-value: "
                  << (agent->sum_q / agent->policy_actions)
                  << std::endl;
    }

    // at this point, we have some playing experience, which we're going to use
    // in order to train the agent.
    relearn::q_probabilistic<state,action> learner;
    for (auto & episode : experience) {
        for (int i = 0; i < 10; i++) {
            learner(episode, policies);
        }
    }
    // clear experience - we'll add new ones!
    experience.clear();
    goto start;

    return 0;
}
