#include <string>
#include "relearn.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("state template class test", "[class_test]") 
{
    auto s_x = relearn::state<int>(0); 
    auto s_y = relearn::state<int>(1., 1);
    REQUIRE(s_x.trait()  == 0);
    REQUIRE(s_y.reward() == 1);
    REQUIRE(s_y.trait() == 1);
    REQUIRE(s_y.reward()== 1);
    REQUIRE(s_x == s_x);
    REQUIRE(s_y == s_y);
    REQUIRE(!(s_x == s_y));
    REQUIRE(s_x < s_y);
    REQUIRE(s_x.trait() < s_y.trait());
    REQUIRE(s_x.hash() == 0);
    REQUIRE(s_y.hash() == 1);
    REQUIRE(s_x < s_y);
}

TEST_CASE("action template class test", "[class_test]")
{
    auto a_x = relearn::action<float>(0);
    auto a_y = relearn::action<float>(0.5);
    REQUIRE(a_x.trait() == 0);
    REQUIRE(a_y.trait() == 0.5);
    REQUIRE(!(a_x == a_y));
    REQUIRE(a_x.trait() < a_y.trait());
    REQUIRE(a_x < a_y);
}

TEST_CASE("link template class test", "[class_test]")
{
    using state  = relearn::state<int>;
    using action = relearn::action<float>;
    auto l_x = relearn::link<state,action>{state(0), action(0)};
    auto l_y = relearn::link<state,action>{state(1), action(1)};
    REQUIRE(l_x == l_x);
    REQUIRE(l_y == l_y);
    REQUIRE(!(l_x == l_y));
    REQUIRE(l_x < l_y);
}

SCENARIO("policy template class test", "[class_test]")
{
    GIVEN("a policy memory of string states and uint actions")
    {
        using state  = relearn::state<std::string>;
        using action = relearn::action<unsigned int>;
        relearn::policy<state,action> memory;

        WHEN("Q-values are updated") {
            memory.update(state("hello"), action(1), 0);
            memory.update(state("world"), action(2), 1);
            THEN("policy returns appropriate values") {
                REQUIRE(memory.value(state("hello"), action(1)) == 0);
                REQUIRE(memory.value(state("world"), action(2)) == 1);
            }
        }

        WHEN("actions are retrieved") {
            memory.update(state("hello"), action(1), 0);
            memory.update(state("world"), action(2), 1);
            auto act_x = memory.actions(state("hello"));
            auto act_y = memory.actions(state("world"));
            THEN("actions match the ones previously inserted") {
                REQUIRE(act_x[action(1)] == 0);
                REQUIRE(act_y[action(2)] == 1);
            }
        }

        WHEN("best action is queried") {
            memory.update(state("hello"), action(1), 0);
            memory.update(state("world"), action(2), 1);
            auto act_y_ptr = memory.best_action(state("world"));
            auto act_y_val = memory.best_value(state("world"));
            THEN("pointer is valid and matches action, and action has Q-value") {
                REQUIRE(*act_y_ptr == action(2));
                REQUIRE(act_y_val  == 1);
            }
            auto act_x_ptr = memory.best_action(state("hello"));
            auto act_x_val = memory.best_value(state("hello"));
            THEN("pointer is valid and matches action, and action has Q-value") {
                REQUIRE(*act_x_ptr == action(1));
                REQUIRE(act_x_val  == 0);
            }
        }
    }

    GIVEN("two policy memories")
    {
        using state  = relearn::state<std::string>;
        using action = relearn::action<unsigned int>;
        relearn::policy<state,action> lhs, rhs;

        WHEN("Q-values are updated in both and concatenated in `lhs`") {
            lhs.update(state("hello"), action(1), 0);
            lhs.update(state("world"), action(2), 1);
            
            rhs.update(state("hello"), action(1), 0);
            rhs.update(state("cruel"), action(2), 0);
            rhs.update(state("world"), action(3), 1);

            lhs += rhs;

            THEN("`lhs` must now contain `rhs` states-actions") {
                REQUIRE(lhs.value(state("hello"), action(1)) == 0);
                REQUIRE(lhs.value(state("cruel"), action(2)) == 0);
                REQUIRE(lhs.value(state("world"), action(3)) == 1);
                REQUIRE(lhs.value(state("world"), action(2)) == 1);
            }
        }

    }
}
