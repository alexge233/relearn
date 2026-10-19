// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include <set>
#include <string>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Regression tests for bugs fixed in the core library.
 * Each test case documents the bug it guards against, so that it never comes back.
 */

/// verifies that `link::operator<` orders links consistently, so that sets and maps keep them all
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

/// verifies that a state whose reward is a `float` can be compared and hashed
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

/// verifies that a learner can train a policy that stores its values as `float`
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
    // deterministic episode: with α = 1 the values are exact
    learner.alpha = 1.f;
    learner(episode, memory);
    learner(episode, memory);
    REQUIRE(memory.value(state("goal"), action("stop")) == Approx(1.f));
    REQUIRE(memory.value(state("start"), action("go")) > 0.f);
}

/// verifies that `policy::operator+=` merges two policies, the right-hand one winning on duplicates
TEST_CASE("policy concatenation", "[regression]")
{
    // previously `operator+=` indexed the maps with the whole key/value pair
    // and failed to compile
    using state  = relearn::state<std::string>;
    using action = relearn::action<unsigned int>;
    relearn::policy<state,action> lhs;
    relearn::policy<state,action> rhs;
    lhs.update(state("hello"), action(1), 0.5);
    rhs.update(state("hello"), action(1), 2.0);
    rhs.update(state("world"), action(2), 1.0);
    lhs += rhs;
    // `rhs` takes precedence on duplicates
    REQUIRE(lhs.value(state("hello"), action(1)) == 2.0);
    REQUIRE(lhs.value(state("world"), action(2)) == 1.0);
}

/// verifies that `q_probabilistic` weighs each next state by how often it was observed
TEST_CASE("q_probabilistic uses transition frequencies", "[regression]")
{
    using state  = relearn::state<std::string>;
    using action = relearn::action<std::string>;
    using link   = relearn::link<state,action>;
    relearn::policy<state,action> memory;
    relearn::q_probabilistic<state,action> learner;
    // (s0, a) leads to s1 (final reward +1) or s2 (final reward -1)
    std::deque<link> to_s1 = {{state("s0"), action("a")},
                              {state(1, "s1"), action("x")}};
    std::deque<link> to_s2 = {{state("s0"), action("a")},
                              {state(-1, "s2"), action("x")}};

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

/// verifies that asking a policy about unknown states or actions never inserts them
TEST_CASE("policy queries do not modify the policy", "[regression]")
{
    // previously every query used `operator[]`, so asking about an unknown
    // state (or action) silently inserted an empty entry into the policy
    using state  = relearn::state<std::string>;
    using action = relearn::action<unsigned int>;

    /**
     * @struct inspect
     * @brief a policy that can count its stored states
     * @note inheriting gives access to the stored policies (see class `policy`)
     */
    struct inspect : relearn::policy<state,action>
    {
        /// @return how many states the policy stores
        std::size_t states() const
        {
            return policies_.size();
        }
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

/// verifies that `terminated = false` learns the last link only as the next state of the one before
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
        // what s1 is already worth
        memory.update(state("s1"), action("b"), 1.0);
        relearn::q_learning<state,action> learner;
        learner.alpha = 1;
        learner.gamma = 0.9;
        learner(episode, memory, false);
        // Q(s0,a) = 0 + 0.9 * Q(s1,b), and Q(s1,b) is unchanged
        REQUIRE(memory.value(state("s0"), action("a")) == Approx(0.9));
        REQUIRE(memory.value(state("s1"), action("b")) == Approx(1.0));
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

/// verifies that the learners accept any episode container offering `size()` and `operator[]`
TEST_CASE("learners only need size() and [] from the episode container", "[regression]")
{
    // the episode type is a template parameter: a container that offers only
    // size() and operator[] (no empty(), no iterators) must still work
    using state  = relearn::state<std::string>;
    using action = relearn::action<std::string>;
    using link   = relearn::link<state,action>;

    /**
     * @struct minimal_chain
     * @brief the smallest episode container the learners accept
     */
    struct minimal_chain
    {
        std::vector<link> links;

        /// @return the number of links
        std::size_t size() const
        {
            return links.size();
        }

        /// @return the link at @param i
        link & operator[](std::size_t i)
        {
            return links[i];
        }

        /// @return the link at @param i
        const link & operator[](std::size_t i) const
        {
            return links[i];
        }
    };
    minimal_chain episode{{{state(0, "s0"), action("a")},
                           {state(1, "s1"), action("b")}}};
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
