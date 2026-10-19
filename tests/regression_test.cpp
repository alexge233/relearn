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

TEST_CASE("policy queries do not modify the policy", "[regression]")
{
    // previously every query used `operator[]`, so asking about an unknown
    // state (or action) silently inserted an empty entry into the policy
    using state  = relearn::state<std::string>;
    using action = relearn::action<unsigned int>;
    // inheriting gives access to the stored policies (see class `policy`)
    struct inspect : relearn::policy<state,action>
    {
        std::size_t states() const { return __policies__.size(); }
    };
    inspect memory;
    memory.update(state("known"), action(1), 0.5);

    // all queries work on a const policy
    const inspect & view = memory;
    REQUIRE(view.value(state("ghost"), action(1)) == 0);
    REQUIRE(std::isnan(view.best_value(state("ghost"))));
    REQUIRE(!view.best_action(state("ghost")));
    REQUIRE(!view.best(state("ghost")).first);
    REQUIRE(view.actions(state("ghost")).empty());
    // unknown action of a known state
    REQUIRE(view.value(state("known"), action(2)) == 0);

    // nothing was inserted
    REQUIRE(view.states() == 1);
    REQUIRE(view.actions(state("known")).size() == 1);
    REQUIRE(view.best_value(state("known")) == 0.5);
}

TEST_CASE("episodes cut short are not learnt as the end of the game", "[regression]")
{
    // an episode stopped by a step limit has not ended: its last link has no
    // outcome yet, so it must only serve as the next state of the link before it
    using state  = relearn::state<std::string>;
    using action = relearn::action<std::string>;
    using link   = relearn::link<state,action>;
    std::deque<link> episode = {{state(0, "s0"), action("a")},
                                {state(5, "s1"), action("b")}};

    SECTION("q_learning") {
        relearn::policy<state,action> memory;
        memory.update(state("s1"), action("b"), 1.0);   // what s1 is already worth
        relearn::q_learning<state,action> learner;
        learner.alpha = 1;
        learner.gamma = 0.9;
        learner(episode, memory, false);
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(0.9));  // 0 + 0.9 * Q(s1)
        REQUIRE(memory.value(state("s1"), action("b")) == Approx(1.0));  // unchanged
        // the same episode, when it really ended, learns its last reward
        learner(episode, memory, true);
        REQUIRE(memory.value(state("s1"), action("b")) == Approx(5.0));
    }
    SECTION("q_probabilistic") {
        relearn::policy<state,action> memory;
        memory.update(state("s1"), action("b"), 1.0);
        relearn::q_probabilistic<state,action> learner(0.9);
        learner(episode, memory, false);
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(0.9));
        REQUIRE(memory.value(state("s1"), action("b")) == Approx(1.0));
        learner(episode, memory, true);
        REQUIRE(memory.value(state("s1"), action("b")) == Approx(5.0));
    }
}

TEST_CASE("learners only need size() and [] from the episode container", "[regression]")
{
    // the episode type is a template parameter: a container that offers only
    // size() and operator[] (no empty(), no iterators) must still work
    using state  = relearn::state<std::string>;
    using action = relearn::action<std::string>;
    using link   = relearn::link<state,action>;
    struct minimal_chain
    {
        std::vector<link> links;
        std::size_t size() const { return links.size(); }
        link & operator[](std::size_t i) { return links[i]; }
        const link & operator[](std::size_t i) const { return links[i]; }
    };
    minimal_chain episode{{{state(0, "s0"), action("a")}, {state(1, "s1"), action("b")}}};
    relearn::policy<state,action> memory;
    relearn::q_learning<state,action,minimal_chain> q;
    q.alpha = 1;
    q(episode, memory);
    q(episode, memory, false);
    relearn::q_probabilistic<state,action,minimal_chain> p(0.9);
    p(episode, memory);
    p(episode, memory, false);
    REQUIRE(memory.value(state("s1"), action("b")) == Approx(1.0));
}
