// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include <set>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Regression tests for bugs fixed in the core library.
 * Each test case documents the bug it guards against.
 */

TEST_CASE("link ordering is a strict weak ordering", "[regression]")
{
    using state  = relearn::state<std::string>;
    using action = relearn::action<std::string>;
    using link   = relearn::link<state,action>;
    // previously `operator<` required *both* state and action to be smaller,
    // so these two distinct links were neither less, greater nor equal
    link x {state("a"), action("b")};
    link y {state("b"), action("a")};
    REQUIRE(!(x == y));
    REQUIRE((x < y) != (y < x));
    REQUIRE(!(x < x));

    // same state, ordered by action
    link z {state("a"), action("c")};
    REQUIRE(x < z);
    REQUIRE(!(z < x));

    // ordered containers must keep all distinct links
    std::set<link> links {x, y, z};
    REQUIRE(links.size() == 3);
}

TEST_CASE("state supports a non-default value_type", "[regression]")
{
    // previously `operator==`/`operator<` took `state<T>` (i.e. `state<T,double>`)
    // so `state<T,float>` could not be compared or hashed
    using state = relearn::state<int,float>;
    state s_x(0);
    state s_y(1.f, 1);
    REQUIRE(s_x == s_x);
    REQUIRE(!(s_x == s_y));
    REQUIRE(s_x < s_y);
    REQUIRE(relearn::hasher<state>{}(s_y) == s_y.hash());
}

TEST_CASE("q_learning works with a float policy", "[regression]")
{
    // previously learners always took `policy<S,A>` (double), ignoring `value_type`
    using state  = relearn::state<std::string,float>;
    using action = relearn::action<std::string>;
    using link   = relearn::link<state,action>;
    relearn::policy<state,action,float> memory;
    std::deque<link> episode = {
        {state("start"), action("go")},
        {state(1.f, "goal"), action("stop")}
    };
    relearn::q_learning<state,action,std::deque<link>,float> learner;
    learner.alpha = 1.f;    // deterministic episode: values are exact
    learner(episode, memory);
    learner(episode, memory);
    REQUIRE(memory.value(state("goal"), action("stop")) == Approx(1.f));
    REQUIRE(memory.value(state("start"), action("go")) > 0.f);
}

TEST_CASE("policy concatenation", "[regression]")
{
    // previously `operator+=` indexed the maps with the whole key/value pair
    // and failed to compile
    using state  = relearn::state<std::string>;
    using action = relearn::action<unsigned int>;
    relearn::policy<state,action> lhs, rhs;
    lhs.update(state("hello"), action(1), 0.5);
    rhs.update(state("hello"), action(1), 2.0);
    rhs.update(state("world"), action(2), 1.0);
    lhs += rhs;
    // `rhs` takes precedence on duplicates
    REQUIRE(lhs.value(state("hello"), action(1)) == 2.0);
    REQUIRE(lhs.value(state("world"), action(2)) == 1.0);
}

TEST_CASE("q_probabilistic uses transition frequencies", "[regression]")
{
    using state  = relearn::state<std::string>;
    using action = relearn::action<std::string>;
    using link   = relearn::link<state,action>;
    relearn::policy<state,action> memory;
    relearn::q_probabilistic<state,action> learner;
    // (s0, a) leads to s1 (final reward +1) or s2 (final reward -1)
    std::deque<link> to_s1 = {{state("s0"), action("a")}, {state(1, "s1"), action("x")}};
    std::deque<link> to_s2 = {{state("s0"), action("a")}, {state(-1, "s2"), action("x")}};

    learner(to_s1, memory);
    learner(to_s1, memory);
    learner(to_s2, memory);
    learner(to_s1, memory);
    // observed transitions: s1 three times, s2 once => P(s1) = 3/4, P(s2) = 1/4
    // previously the count was always 1 (`=+ 1`) and the probability used
    // integer division over distinct successors, so this was 1/2 => 0
    // Q(s0,a) = r(s0) + gamma * (P(s1) * maxQ(s1) + P(s2) * maxQ(s2))
    //         = 0 + 0.9 * (0.75 * 1 + 0.25 * -1)
    REQUIRE(memory.value(state("s0"), action("a")) == Approx(0.9 * 0.5));
}
