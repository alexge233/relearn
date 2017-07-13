#include "relearn.hpp"
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("state class serialization test", "[state_class_serialization_test]") 
{
    auto s_x = relearn::state<std::string>("hello");
    auto s_y = relearn::state<std::string>("world");
    // create and open a character archive for output
    std::ofstream ofs("serialize_test_state_class");
    boost::archive::text_oarchive oa(ofs);
    // write class instance to archive
    oa << s_x;
    ofs.close();
    // load data from the archive now
    std::ifstream ifs("serialize_test_state_class");
    boost::archive::text_iarchive ia(ifs);
    ia >> s_y;
    // if serialization failed, then s_x will not equal s_y!
    REQUIRE(s_x == s_y);
}

TEST_CASE("action class serialization test", "[action_class_serialization_test]") 
{
    auto a_x = relearn::action<int>(0);
    auto a_y = relearn::action<int>(1);
    // create and open a character archive for output
    std::ofstream ofs("serialize_test_action_class");
    boost::archive::text_oarchive oa(ofs);
    // write class instance to archive
    oa << a_x;
    ofs.close();
    // load data from the archive now
    std::ifstream ifs("serialize_test_action_class");
    boost::archive::text_iarchive ia(ifs);
    ia >> a_y;
    // if serialization failed, then a_x will not equal a_y!
    REQUIRE(a_x == a_y);
}

SCENARIO("policy class serialization test", "[policy_class_serialize_test]")
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
        WHEN("policy is trained and saved to disk") {
            auto learner = relearn::q_learning<state,action>{0.9, 0.9};
            for (int k = 0; k < 10; k++) {
                learner(episode, memory);
            }
            std::ofstream ofs("serialize_test_policy_class");
            boost::archive::text_oarchive oa(ofs);
            oa << memory;
            ofs.close();
            THEN("we load a new policy and expect the same output") {
                relearn::policy<state,action> policies;
                std::ifstream ifs("serialize_test_policy_class");
                boost::archive::text_iarchive ia(ifs);
                ia >> policies;
                REQUIRE(policies.best_action(state("hello"))->trait() == "hi!");
                REQUIRE(policies.best_action(state("how are you?"))->trait() == "I'm fine, and you?");
                REQUIRE(policies.best_action(state("not too bad! what you doing here?"))->trait()
                        == "I'm taking over the world!");
            }
        }
    }
}
