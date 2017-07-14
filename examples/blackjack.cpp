/**
 * A Blackjack/21 example showing how non-deterministic (probabilistic)
 * Episodic Q-Learning works.
 * 
 * @version 0.1.0
 * @author Alex Giokas
 * @date 19.11.2016
 *
 * @see the header (blackjack_header.hpp) for implementation details
 */
#include "blackjack_header.hpp"
#include <boost/predef.h>

// create aliases for state and action: 
// - a state is the current hand held by a player
// - an action is draw(true) or stay(false)
//
using state = relearn::state<hand>;
using action = relearn::action<bool>;
// policy memory
relearn::policy<state,action> policies;

// catching CTRL-C works only on Linux for now
#if BOOST_OS_LINUX
#include <signal.h>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
volatile sig_atomic_t flag = 0;
void signal_handler(int sig) {
    flag = 1;
}
#endif

// TODO: add BOOST_OS_WINDOWS
//       and BOOST_OS_MACOS

int main()
{
    signal(SIGINT, signal_handler);
    // PRNG needed for, magic random voodoo
    std::mt19937 gen(static_cast<std::size_t>(std::chrono::high_resolution_clock::now()
                                                           .time_since_epoch().count()));
    // create the dealer, and two players...
    auto dealer = std::make_shared<house>(cards, gen);
    auto agent = std::make_shared<client>();

    using link = relearn::link<state,action>;
    std::deque<std::deque<link>>  experience;
    
    float sum  = 0;
    float wins = 0;
    std::cout << "starting! Press CTRL-C to stop at any time!" 
              << std::endl;
    start:
    // play 10 rounds - then stop
    for (int i = 0; i < 100; i++) {
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
    }

    // CTRL-C/SIGINT => save and exit
    if (flag) {
    #if USING_BOOST_SERIALIZATION
        std::cout << "save & exit\r\n";
        std::ofstream ofs("blackjack.policy");
        boost::archive::text_oarchive oa(ofs);
        oa << policies;
        ofs.close();
        return 0;
    #else
        std::cout << "exiting...\r\n";
    #endif
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
    agent->reset();
    // restart - please don't use goto's in modern C++
    goto start;

    return 0;
}
