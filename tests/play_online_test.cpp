// `relearn.hpp` is deliberately included first: it must compile on its own
// without relying on headers pulled in by the includer.
#include "relearn.hpp"
#include "../examples/games/game.hpp"
#include <string>
#include <vector>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

/**
 * Tests of `games::play_online`, which every learner that learns after each step depends on:
 * exactly what it gives the learner, and when. The game is a scripted corridor, and the learner
 * only records what it is given, so that every call can be checked.
 */

/**
 * @class corridor
 * @brief cells 0, 1, 2, 3: the one action of cell c, numbered 10 + c, moves one cell to the right, earning
 *        the number of the cell reached, and reaching cell 3 ends the game. Each cell's action has its own
 *        number, so that the tests can tell which action was given with which link
 */
class corridor
{
public:
    using state_type  = unsigned int;
    using action_type = unsigned int;

    /// @brief start again @return cell 0
    unsigned int reset()
    {
        at_ = 0;
        return at_;
    }

    /// @brief move right @return the cell reached, its reward, and whether the game ended
    games::step_result<unsigned int> step(const unsigned int &)
    {
        at_++;
        games::step_result<unsigned int> out;
        out.next = at_;
        out.reward = at_;
        out.terminated = at_ == 3;
        return out;
    }

    /// @return the single action of cell @param s, 10 + s
    std::vector<unsigned int> actions(const unsigned int & s) const
    {
        return {10 + s};
    }

private:
    unsigned int at_ = 0;
};

using types = games::types<corridor>;

/**
 * @struct recorder
 * @brief a learner that learns nothing, and records every call it is given, in the shared @param log
 */
struct recorder
{
    /// the calls so far, as text: "learn [(s, a, r)(s, a, r)] terminated" or "choose s"
    std::vector<std::string> & log;

    void operator()(types::episode episode,
                    types::policy &,
                    bool terminated)
    {
        std::string line = "learn [";
        for (const auto & link : episode) {
            line += "(" + std::to_string(link.state.trait()) + ", " + std::to_string(link.action.trait()) + ", "
                  + std::to_string(static_cast<int>(link.state.reward())) + ")";
        }
        line += terminated ? "] terminated" : "] cut";
        log.push_back(line);
    }
};

/**
 * @struct starter
 * @brief the same, for a learner that keeps something from one call to the next, and is told when
 *        a new episode starts (as `relearn::true_online_sarsa_lambda`): it records "start" too
 */
struct starter : recorder
{
    /// @brief record into @param shared
    explicit starter(std::vector<std::string> & shared)
    : recorder{shared}
    {}

    void start_episode()
    {
        log.push_back("start");
    }
};

/// verifies the calls of a whole game: each step learnt once, from the next state and action chosen before learning
TEST_CASE("play_online: what the learner is given, and when", "[play_online]")
{
    std::vector<std::string> log;
    recorder learner{log};
    types::policy memory;
    corridor game;
    auto agent = [&log](const unsigned int & s,
                        const std::vector<unsigned int> &) {
        log.push_back("choose " + std::to_string(s));
        return 10 + s;
    };
    auto r = games::play_online(game, agent, learner, memory, 100);
    // the first action is chosen in cell 0; then after each step the next action is chosen first
    // (SARSA learns step t from it), and step t is learnt from the state reached and *that* action,
    // not yet as an end: its second link carries no reward of its own. The last step is learnt alone
    const std::vector<std::string> expected = {
        "choose 0",
        "choose 1", "learn [(0, 10, 1)(1, 11, 0)] cut",
        "choose 2", "learn [(1, 11, 2)(2, 12, 0)] cut",
        "learn [(2, 12, 3)] terminated"
    };
    REQUIRE(log == expected);
    REQUIRE(r.terminated);
    REQUIRE(r.steps == 3);
    REQUIRE(r.total_reward == 6);
    // the episode played: every step once, with its reward, and nothing else
    REQUIRE(r.episode.size() == 3);
    for (unsigned int t = 0; t < 3; t++) {
        REQUIRE(r.episode[t].state.trait() == t);
        REQUIRE(r.episode[t].action.trait() == 10 + t);
        REQUIRE(r.episode[t].state.reward() == t + 1);
    }
}

/// verifies a game cut short: no step is learnt as an end, and the episode ends with the state reached
TEST_CASE("play_online: a game cut short", "[play_online]")
{
    std::vector<std::string> log;
    recorder learner{log};
    types::policy memory;
    corridor game;
    auto agent = [&log](const unsigned int & s,
                        const std::vector<unsigned int> &) {
        log.push_back("choose " + std::to_string(s));
        return 10 + s;
    };
    auto r = games::play_online(game, agent, learner, memory, 2);
    const std::vector<std::string> expected = {
        "choose 0",
        "choose 1", "learn [(0, 10, 1)(1, 11, 0)] cut",
        "choose 2", "learn [(1, 11, 2)(2, 12, 0)] cut"
    };
    REQUIRE(log == expected);
    REQUIRE_FALSE(r.terminated);
    REQUIRE(r.steps == 2);
    // as with `play`, the episode ends with one extra link: the state reached and the action chosen next,
    // which is not a step (its reward is 0)
    REQUIRE(r.episode.size() == 3);
    REQUIRE(r.episode.back().state.trait() == 2);
    REQUIRE(r.episode.back().action.trait() == 12);
    REQUIRE(r.episode.back().state.reward() == 0);
    for (unsigned int t = 0; t < 2; t++) {
        REQUIRE(r.episode[t].action.trait() == 10 + t);
    }
}

/// verifies that a learner with a start_episode() is told before every game, even after one cut short
TEST_CASE("play_online: a learner that keeps its traces is told when a game starts", "[play_online]")
{
    std::vector<std::string> log;
    starter learner(log);
    types::policy memory;
    corridor game;
    auto agent = [](const unsigned int & s,
                    const std::vector<unsigned int> &) {
        return 10 + s;
    };
    games::play_online(game, agent, learner, memory, 2);
    games::play_online(game, agent, learner, memory, 100);
    const std::vector<std::string> expected = {
        "start", "learn [(0, 10, 1)(1, 11, 0)] cut", "learn [(1, 11, 2)(2, 12, 0)] cut",
        "start", "learn [(0, 10, 1)(1, 11, 0)] cut", "learn [(1, 11, 2)(2, 12, 0)] cut", "learn [(2, 12, 3)] terminated"
    };
    REQUIRE(log == expected);
}
