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

/// pins how relearn takes max Q(s', a): over the actions stored for s' only (have a look at docs/q_learning.md)
TEST_CASE("q_learning: the maximum counts only the actions stored", "[regression]")
{
    using state  = relearn::state<std::string>;
    using action = relearn::action<std::string>;
    using link   = relearn::link<state,action>;
    // α = 1, γ = 0.9. In s1, only y is stored, worth -1; x has never been tried there.
    // (s0, a) earns 0 and leads to s1: Q(s0, a) = 0 + 0.9 * max Q(s1, .) = 0.9 * -1 = -0.9,
    // where the book, which starts x at 0, would learn 0.9 * max(0, -1) = 0
    relearn::policy<state,action> memory;
    memory.update(state("s1"), action("y"), -1);
    relearn::q_learning<state,action> learner(1.0, 0.9);
    std::deque<link> episode = {{state(0, "s0"), action("a")},
                                {state(0, "s1"), action("y")}};
    learner(episode, memory, false);
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(-0.9));
    // a state with nothing stored counts 0
    relearn::policy<state,action> empty;
    learner(episode, empty, false);
    REQUIRE(empty.value(state("s0"), action("a")) == Approx(0));
}
