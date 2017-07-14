#include <string>
#include "relearn.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

SCENARIO("Deterministic Q-Learning", "[train_test_1]")
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
        WHEN("q-learning is used to train") {
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

SCENARIO("non-deterministic Q-probabilistic", "[train_test_2]")
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
        WHEN("q-probabilistic is used to train") {
            auto learner = relearn::q_probabilistic<state,action>();
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
}
