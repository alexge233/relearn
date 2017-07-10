#include <string>
#include "relearn.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// TODO: create a LOGIC test, where we manually populate a determined list
//       of episodes (e.g., (0,0), (0,1), (0,2), (0,3), (0,4)
//       with their appropriate actions (e.g., N, E, W, S)
//       and a respective Reward
//       Repeat a few times (2-3 times is enough) with a positive and negative reward
//       then train the algorithm and policies,
//       and finally hard-core the policies and Q-Values to make sense
//       this way we can logically validate that the algorithm works fine
//       without issues or undefined behaviour
//

SCENARIO("Deterministic Q-Learning Test #1", "[train_test_1]")
{
    using state  = relearn::state<std::string>;
    using action = relearn::action<std::string>;
    using link   = relearn::link<state,action>;
    relearn::policy<state,action> memory;

    GIVEN("hard-coded episode with positive reward")
    {
        std::deque<link> episode = {
            {state("hello"), action("hi!")},
            {state("how are you?"), action("I'm fine, and you?")},
            {state(1, "not too bad! what you doing here?"), 
             action("I'm taking over the world!")},
        };
        WHEN("Q-learning is used to train") {
            auto learner = relearn::q_learning<state,action>{0.9, 0.9};
            for (int k = 0; k < 10; k++) {
                learner(episode, memory);
            }
            THEN("we expect the episode to be replayed") {
                REQUIRE(memory.best_action(state("hello"))->trait() == "hi!");
                REQUIRE(memory.best_action(state("how are you?"))->trait() == "I'm fine, and you?");
                REQUIRE(memory.best_action(state("not too bad! what you doing here?"))->trait()
                        == "I'm taking over the world!");
            }
        }
    }
    GIVEN("hard-coded episode with negative reward")
    {
        std::deque<link> episode = {
            {state("hello"), action("what do you want?")},
            {state("how are you?"), action("why do you care?")},
            {state(-1, "wow, sorry for asking!"), action("piss off!")}
        };
        WHEN("Q-learning is used to train") {
            auto learner = relearn::q_learning<state,action>{0.9, 0.9};
            for (int k = 0; k < 10; k++) {
                learner(episode, memory);
            }
            THEN("we expect episode's actions to have negative Q-values") {
                REQUIRE(memory.value(state("hello"), action("what do you want?")) < 0);
                REQUIRE(memory.value(state("how are you?"), action("why do you care?")) < 0);
                REQUIRE(memory.value(state("wow, sorry for asking!"), action("piss off!")) < 0);
            }
        }
    }
}

// TODO: create a similar test for Q-Probabilistic?
