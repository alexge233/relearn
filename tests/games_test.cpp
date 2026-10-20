/**
 * Tests of the example games: their rules, and that relearn's learners reach
 * the known answer for each one (exact solutions from `solvers.hpp`, or the
 * optimal path lengths given by Sutton & Barto).
 * Random number generators are seeded, so results are reproducible.
 */
#include "../examples/games/blackjack.hpp"
#include "../examples/games/cliff_walking.hpp"
#include "../examples/games/windy_gridworld.hpp"
#include "../examples/games/frozen_lake.hpp"
#include "../examples/games/bandit.hpp"
#include "../examples/games/random_walk.hpp"
#include "../examples/games/short_corridor.hpp"
#include "../examples/games/maximization_bias.hpp"
#include "../examples/games/gamblers_problem.hpp"
#include "../examples/games/dyna_maze.hpp"
#include "../examples/games/tic_tac_toe.hpp"
#include "../examples/games/cart_pole.hpp"
#include "../examples/games/mountain_car.hpp"
#include "../examples/games/discretised.hpp"
#include <cmath>
#include <map>
#include <queue>
#include "solvers.hpp"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

namespace
{
/**
 * @brief train a policy on a game, exploring with ε-greedy
 * @param game is the game to play
 * @param learner updates the policy after each episode
 * @param memory is the policy being learnt
 * @param episodes is the number of episodes to play
 * @param epsilon is the probability of a random move
 * @param max_steps cuts an episode short
 * @param prng is the random number generator of the agent
 */
template <class game_type,
          class learner_type>
void train(game_type & game,
           learner_type & learner,
           typename games::types<game_type>::policy & memory,
           unsigned int episodes,
           double epsilon,
           std::size_t max_steps,
           std::mt19937 & prng)
{
    // the ε-greedy agent: explores with probability ε, otherwise takes the best known move
    auto agent = [&](const typename game_type::state_type & s,
                     const std::vector<typename game_type::action_type> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, epsilon, prng);
    };
    for (unsigned int i = 0; i < episodes; i++) {
        // play one episode, then learn from it
        auto r = games::play(game, agent, max_steps);
        learner(r.episode, memory, r.terminated);
    }
}
} // namespace

// ------------------------------------------------------------------ blackjack

/// verifies how a hand counts its Aces: 11 when that does not bust it, 1 otherwise
TEST_CASE("blackjack: hand totals", "[games][blackjack]")
{
    // two Aces: one counts 11, the other 1
    games::blackjack_hand two_aces;
    two_aces.add(1);
    two_aces.add(1);
    REQUIRE(two_aces.sum == 12);
    REQUIRE(two_aces.usable_ace);

    games::blackjack_hand natural;
    natural.add(1);
    natural.add(10);
    REQUIRE(natural.natural());

    // A + 5 + 9: the Ace must drop back to 1
    games::blackjack_hand soft;
    soft.add(1);
    soft.add(5);
    REQUIRE(soft.sum == 16);
    REQUIRE(soft.usable_ace);
    soft.add(9);
    REQUIRE(soft.sum == 15);
    REQUIRE(!soft.usable_ace);
    REQUIRE(!soft.bust());
    REQUIRE(!soft.natural());
}

/// verifies the flow of a game: decisions start at 12, hitting busts at -1, sticking ends the game
TEST_CASE("blackjack: game flow", "[games][blackjack]")
{
    games::blackjack game(7);
    for (int i = 0; i < 1000; i++) {
        auto s = game.reset();
        REQUIRE(s.player_sum >= 12);
        REQUIRE(s.player_sum <= 21);
        REQUIRE(s.dealer_showing >= 1);
        REQUIRE(s.dealer_showing <= 10);
        // hit until the game ends: busting is -1, otherwise the game continues
        for (;;) {
            auto out = game.step(games::blackjack::hit);
            if (out.terminated) {
                REQUIRE(out.reward == -1);
                break;
            }
            REQUIRE(out.reward == 0);
            REQUIRE(!out.next.natural);
        }
        // sticking always ends the game with -1, 0 or +1
        game.reset();
        auto out = game.step(games::blackjack::stick);
        REQUIRE(out.terminated);
        REQUIRE((out.reward == -1 || out.reward == 0 || out.reward == 1));
    }
}

/// verifies against the exact solver that the learnt strategy is optimal (Sutton & Barto, Figure 5.2)
TEST_CASE("blackjack: q_probabilistic learns the optimal strategy", "[games][blackjack]")
{
    using game_type = games::blackjack;
    using types     = games::types<game_type>;
    game_type game(1);
    std::mt19937 prng(2);
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(1.0);
    train(game, learner, memory, 2000000, 0.1, 100, prng);

    solvers::blackjack exact;
    // the learnt decision matches the optimal one in almost every state; the few
    // exceptions are near-ties, where sampling noise in the estimates (a stick is
    // worth -1, 0 or +1) can put the two actions in the wrong order
    int disagreements = 0;
    for (int usable = 0; usable < 2; usable++) {
        for (unsigned int sum = 12; sum <= 21; sum++) {
            for (unsigned int dealer = 1; dealer <= 10; dealer++) {
                game_type::state_type s;
                s.player_sum     = sum;
                s.usable_ace     = usable == 1;
                s.dealer_showing = dealer;
                // the learnt values, and the exact ones
                double q_stick    = memory.value(types::state(s), types::action(game_type::stick));
                double q_hit      = memory.value(types::state(s), types::action(game_type::hit));
                double best_stick = exact.stick(sum, dealer);
                double best_hit   = exact.hit(sum, usable == 1, dealer);
                if ((q_stick >= q_hit) != (best_stick >= best_hit)) {
                    disagreements++;
                    INFO("sum " << sum << " usable " << usable << " dealer " << dealer);
                    REQUIRE(std::fabs(best_stick - best_hit) < 0.1);
                }
            }
        }
    }
    // of 200 states
    REQUIRE(disagreements <= 10);

    // and it earns what the optimal strategy earns
    auto optimal = [&](const game_type::state_type & s,
                       const std::vector<unsigned int> &) {
        double stick = s.natural ? exact.stick_natural(s.dealer_showing)
                                 : exact.stick(s.player_sum, s.dealer_showing);
        double hit = exact.hit(s.player_sum, s.usable_ace, s.dealer_showing);
        return stick >= hit ? game_type::stick : game_type::hit;
    };
    const unsigned int games_played = 200000;
    game_type eval_learnt(11);
    game_type eval_optimal(11);
    double learnt = games::evaluate(eval_learnt, memory, games_played, 100, prng);
    double best = 0;
    for (unsigned int i = 0; i < games_played; i++) {
        best += games::play(eval_optimal, optimal, 100).total_reward;
    }
    best /= games_played;
    INFO("learnt " << learnt << " optimal " << best);
    REQUIRE(std::fabs(learnt - best) < 0.01);
}

// -------------------------------------------------------------- cliff walking

/// verifies the rules of Cliff Walking (Sutton & Barto, Example 6.6)
TEST_CASE("cliff walking: rules", "[games][cliff]")
{
    games::cliff_walking game;
    auto s = game.reset();
    REQUIRE(s == games::cliff_walking::start());
    // the border keeps the agent in place, at the usual cost
    auto out = game.step(games::left);
    REQUIRE(out.next == games::cliff_walking::start());
    REQUIRE(out.reward == -1);
    REQUIRE(!out.terminated);
    // the cliff costs -100 and sends the agent back to the start, without ending
    out = game.step(games::right);
    REQUIRE(out.reward == -100);
    REQUIRE(out.next == games::cliff_walking::start());
    REQUIRE(!out.terminated);
    // walking around the cliff reaches the goal in 13 moves
    game.reset();
    std::vector<unsigned int> path(1, games::up);
    path.insert(path.end(), 11, games::right);
    path.push_back(games::down);
    double total = 0;
    for (std::size_t i = 0; i < path.size(); i++) {
        out = game.step(path[i]);
        total += out.reward;
        REQUIRE(out.terminated == (i + 1 == path.size()));
    }
    REQUIRE(out.next == games::cliff_walking::goal());
    REQUIRE(total == -13);
}

/// verifies that Q-learning learns the 13-move path along the cliff edge
TEST_CASE("cliff walking: q_learning learns the shortest path", "[games][cliff]")
{
    using game_type = games::cliff_walking;
    using types     = games::types<game_type>;
    games::cliff_walking game;
    std::mt19937 prng(1);
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.5;
    learner.gamma = 1.0;
    train(game, learner, memory, 3000, 0.1, 1000, prng);
    // play once greedily: the learnt path should be the shortest
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::greedy<game_type>(memory, s, legal, prng);
    };
    auto greedy = games::play(game, agent, 100);
    REQUIRE(greedy.terminated);
    REQUIRE(greedy.steps == 13);
    REQUIRE(greedy.total_reward == -13);
}

// ------------------------------------------------------------ windy gridworld

namespace
{
/**
 * @brief the exact model of the windy gridworld, written from the rules of Sutton & Barto
 * @param stochastic is true for random wind (Exercise 6.10)
 * @return the model, one state per cell (s = y * width + x)
 */
solvers::tabular windy_model(bool stochastic)
{
    using game_type = games::windy_gridworld;
    solvers::tabular m;
    m.states  = game_type::width * game_type::height;
    m.actions = 4;
    m.gamma   = 1;
    m.model = [stochastic](int s, int a) {
        std::vector<solvers::outcome> out;
        int x = s % game_type::width;
        int y = s / game_type::width;
        // the goal ends the episode: no moves from it
        if (x == 7 && y == 3) {
            return out;
        }
        // the wind of each column pushes up; random wind adds -1, 0 or +1 to it
        const int wind[] = {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};
        std::vector<int> pushes = {wind[x]};
        if (stochastic && wind[x] > 0) {
            pushes = {wind[x] - 1, wind[x], wind[x] + 1};
        }
        for (int push : pushes) {
            // the move (0 up, 1 right, 2 down, 3 left), kept inside the grid
            int nx = std::min(std::max(x + (a == 1) - (a == 3), 0), game_type::width - 1);
            int ny = std::min(std::max(y - (a == 0) + (a == 2), 0), game_type::height - 1);
            // then the wind, also kept inside the grid
            ny = std::min(std::max(ny - push, 0), game_type::height - 1);
            out.push_back({1.0 / pushes.size(), ny * game_type::width + nx, -1.0, nx == 7 && ny == 3});
        }
        return out;
    };
    return m;
}

/**
 * @brief the greedy action of a policy in every cell of a grid game
 * @param memory is the learnt policy
 * @param width and @param height are the size of the grid
 * @param prng breaks ties between actions of equal value
 * @return one action per cell (s = y * width + x)
 */
template <class game_type>
std::vector<int> greedy_policy(const typename games::types<game_type>::policy & memory,
                               int width,
                               int height,
                               std::mt19937 & prng)
{
    std::vector<int> policy(width * height);
    for (int s = 0; s < width * height; s++) {
        games::position p{s % width, s / width};
        policy[s] = games::greedy<game_type>(memory, p, {0, 1, 2, 3}, prng);
    }
    return policy;
}
} // namespace

/// verifies every move from every cell against the exact model (Sutton & Barto, Example 6.5)
TEST_CASE("windy gridworld: rules", "[games][windy]")
{
    using game_type = games::windy_gridworld;
    game_type game;
    REQUIRE(game.reset() == game_type::start());
    // the game agrees with the exact model on every move from every cell
    auto model = windy_model(false);
    for (int s = 0; s < model.states; s++) {
        games::position at{s % game_type::width, s / game_type::width};
        if (at == game_type::goal()) {
            continue;
        }
        for (unsigned int a = 0; a < 4; a++) {
            game.place(at);
            auto out = game.step(a);
            auto expected = model.model(s, static_cast<int>(a));
            REQUIRE(expected.size() == 1);
            REQUIRE(out.next.x + out.next.y * game_type::width == expected[0].next);
            REQUIRE(out.reward == expected[0].reward);
            REQUIRE(out.terminated == expected[0].terminal);
        }
    }
    // from the start, moving right is pushed up once it leaves column 3 (wind 1)
    game.reset();
    for (int i = 0; i < 3; i++) {
        REQUIRE(game.step(games::right).next == (games::position{i + 1, 3}));
    }
    REQUIRE(game.step(games::right).next == (games::position{4, 2}));
    // random wind only varies where there is wind: column 0 has none
    game_type random_wind(true, 3);
    for (int i = 0; i < 100; i++) {
        random_wind.place({0, 3});
        REQUIRE(random_wind.step(games::up).next == (games::position{0, 2}));
    }
}

/// verifies that Q-learning learns the 15-move path of Sutton & Barto, Example 6.5
TEST_CASE("windy gridworld: q_learning learns the shortest path", "[games][windy]")
{
    using game_type = games::windy_gridworld;
    using types     = games::types<game_type>;
    game_type game;
    std::mt19937 prng(1);
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.5;
    learner.gamma = 1.0;
    train(game, learner, memory, 500, 0.1, 1000, prng);
    // the exact model agrees that the shortest path takes 15 moves
    auto exact = windy_model(false);
    REQUIRE(exact.optimal_values()[3 * game_type::width] == Approx(-15));
    // play once greedily: the learnt path should be the shortest
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::greedy<game_type>(memory, s, legal, prng);
    };
    auto greedy = games::play(game, agent, 100);
    REQUIRE(greedy.terminated);
    REQUIRE(greedy.steps == 15);
}

/// verifies, by exact evaluation, that the policy learnt in random wind (Exercise 6.10) is optimal
TEST_CASE("windy gridworld: q_probabilistic learns the best policy in random wind", "[games][windy]")
{
    using game_type = games::windy_gridworld;
    using types     = games::types<game_type>;
    game_type game(true, 1);
    std::mt19937 prng(11);
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(1.0);
    train(game, learner, memory, 2000, 0.1, 1000, prng);
    // the learnt greedy policy, evaluated exactly, needs as few moves as the optimal one
    auto exact  = windy_model(true);
    auto best   = exact.optimal_values();
    auto learnt = exact.policy_values(greedy_policy<game_type>(memory, game_type::width, game_type::height, prng));
    const int start = 3 * game_type::width;
    REQUIRE(learnt[start] == Approx(best[start]).margin(0.1));
}

/// verifies that, in random wind, `q_learning` with a learning rate that decays per visit
/// (α = 1/n^0.6) learns the best policy, which a constant α = 0.5 does not
TEST_CASE("windy gridworld: q_learning with a decaying learning rate learns the best policy in random wind", "[games][windy][schedules]")
{
    using game_type = games::windy_gridworld;
    using types     = games::types<game_type>;
    game_type game(true, 1);
    std::mt19937 prng(11);
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.gamma = 1.0;
    // α = 1/n^0.6 for the n-th update of each pair: it decays slowly enough to keep
    // learning, and fast enough to average out the wind. α = 1/n decays too fast:
    // its early updates, made from wrong values, weigh as much as the later ones
    learner.learning_rate = relearn::polynomial_decay<>(1.0, 0.6);
    // across seeds 1 ... 10, after 5000 episodes, the learnt policy was the best one
    // in every run (exactly: the gap was below 1e-11)
    train(game, learner, memory, 5000, 0.1, 1000, prng);
    auto exact  = windy_model(true);
    auto best   = exact.optimal_values();
    auto learnt = exact.policy_values(greedy_policy<game_type>(memory, game_type::width, game_type::height, prng));
    const int start = 3 * game_type::width;
    REQUIRE(learnt[start] == Approx(best[start]).margin(0.1));
}

// ---------------------------------------------------------------- frozen lake

namespace
{
/**
 * @brief the exact model of FrozenLake, written from Gymnasium's rules
 * @param map is the lake, one string per row ('S' start, 'F' ice, 'H' hole, 'G' goal)
 * @param gamma is the discount rate
 * @param slippery is true if a move can slide to either side
 * @return the model, one state per cell (s = row * columns + column)
 */
solvers::tabular frozen_lake_model(const std::vector<std::string> & map,
                                   double gamma,
                                   bool slippery)
{
    solvers::tabular m;
    const int rows = static_cast<int>(map.size());
    const int cols = static_cast<int>(map[0].size());
    m.states  = rows * cols;
    m.actions = 4;
    m.gamma   = gamma;
    m.model = [map, rows, cols, slippery](int s, int a) {
        std::vector<solvers::outcome> out;
        // holes and the goal end the episode: no moves from them
        char here = map[s / cols][s % cols];
        if (here == 'H' || here == 'G') {
            return out;
        }
        // on slippery ice, the move goes as intended or slides to either side
        std::vector<int> slides = slippery ? std::vector<int>{-1, 0, 1} : std::vector<int>{0};
        for (int d : slides) {
            int b = (a + 4 + d) % 4;
            int r = s / cols;
            int c = s % cols;
            // Gymnasium's moves: 0 left, 1 down, 2 right, 3 up, kept inside the lake
            if (b == 0) {
                c = std::max(c - 1, 0);
            }
            if (b == 1) {
                r = std::min(r + 1, rows - 1);
            }
            if (b == 2) {
                c = std::min(c + 1, cols - 1);
            }
            if (b == 3) {
                r = std::max(r - 1, 0);
            }
            // reaching the goal is worth 1; the goal and the holes end the episode
            char t = map[r][c];
            out.push_back({1.0 / slides.size(), r * cols + c, t == 'G' ? 1.0 : 0.0, t == 'G' || t == 'H'});
        }
        return out;
    };
    return m;
}
} // namespace

/// verifies every move against the exact model, and that slippery ice slides a third of the time each way
TEST_CASE("frozen lake: rules", "[games][frozen_lake]")
{
    using game_type = games::frozen_lake;
    // without slipping, every move from every cell matches the exact model
    game_type ice(game_type::map_4x4(), false);
    auto model = frozen_lake_model(game_type::map_4x4(), 1.0, false);
    REQUIRE(ice.reset() == 0u);
    REQUIRE(ice.tile(15) == 'G');
    for (int s = 0; s < model.states; s++) {
        for (unsigned int a = 0; a < 4; a++) {
            auto expected = model.model(s, static_cast<int>(a));
            // a hole or the goal
            if (expected.empty()) {
                continue;
            }
            ice.place(static_cast<unsigned int>(s));
            auto out = ice.step(a);
            REQUIRE(static_cast<int>(out.next) == expected[0].next);
            REQUIRE(out.reward == expected[0].reward);
            REQUIRE(out.terminated == expected[0].terminal);
        }
    }
    // on slippery ice, "down" from cell 0 goes down, left (stays) or right, a third each
    game_type slippery(game_type::map_4x4(), true, 5);
    int down  = 0;
    int stay  = 0;
    int right = 0;
    const int n = 30000;
    for (int i = 0; i < n; i++) {
        slippery.place(0);
        auto next = slippery.step(game_type::down).next;
        if (next == 4) {
            down++;
        }
        else if (next == 0) {
            stay++;
        }
        else if (next == 1) {
            right++;
        }
    }
    REQUIRE(down + stay + right == n);
    REQUIRE(down  == Approx(n / 3.0).epsilon(0.05));
    REQUIRE(stay  == Approx(n / 3.0).epsilon(0.05));
    REQUIRE(right == Approx(n / 3.0).epsilon(0.05));
}

/// verifies, by exact evaluation, that the policy learnt on slippery 4x4 ice is optimal
TEST_CASE("frozen lake: q_probabilistic learns the optimal policy", "[games][frozen_lake]")
{
    using game_type = games::frozen_lake;
    using types     = games::types<game_type>;
    const double gamma = 0.95;
    game_type game(game_type::map_4x4(), true, 1);
    std::mt19937 prng(11);
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(gamma);
    train(game, learner, memory, 10000, 0.1, game.step_limit(), prng);
    // the learnt greedy policy, evaluated exactly, is worth as much as the optimal one
    auto exact = frozen_lake_model(game_type::map_4x4(), gamma, true);
    std::vector<int> policy(exact.states);
    for (int s = 0; s < exact.states; s++) {
        policy[s] = static_cast<int>(games::greedy<game_type>(memory, static_cast<unsigned int>(s), {0, 1, 2, 3}, prng));
    }
    double best   = exact.optimal_values()[0];
    double learnt = exact.policy_values(policy)[0];
    REQUIRE(learnt == Approx(best).margin(0.005));
}

// --------------------------------------------------------------------- bandit

/// verifies that, with one-step episodes, `q_probabilistic` stores the sample average of each arm
TEST_CASE("bandit: q_probabilistic keeps the exact average reward of each arm", "[games][bandit]")
{
    using game_type = games::bandit;
    using types     = games::types<game_type>;
    game_type game(10, 3);
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(1.0);
    std::vector<double> sum(10, 0);
    std::vector<double> count(10, 0);
    std::mt19937 prng(4);
    // an agent that pulls any arm at random
    auto any_arm = [&](const unsigned int &,
                       const std::vector<unsigned int> & arms) {
        std::uniform_int_distribution<std::size_t> any(0, arms.size() - 1);
        return arms[any(prng)];
    };
    for (int i = 0; i < 2000; i++) {
        auto r = games::play(game, any_arm, 1);
        REQUIRE(r.terminated);
        REQUIRE(r.steps == 1u);
        // keep our own sample average of each arm
        sum[r.episode[0].action.trait()] += r.total_reward;
        count[r.episode[0].action.trait()] += 1;
        learner(r.episode, memory, r.terminated);
    }
    for (unsigned int a = 0; a < 10; a++) {
        REQUIRE(memory.value(types::state(0), types::action(a)) == Approx(sum[a] / count[a]));
        // and the sample average is close to the arm's true mean
        REQUIRE(sum[a] / count[a] == Approx(game.mean(a)).margin(0.3));
    }
}

/// verifies that ε-greedy finds the best arm in most of 200 bandits (Sutton & Barto, Section 2.3)
TEST_CASE("bandit: epsilon-greedy finds the best arm", "[games][bandit]")
{
    using game_type = games::bandit;
    using types     = games::types<game_type>;
    int found = 0;
    double regret = 0;
    const int runs = 200;
    for (int run = 0; run < runs; run++) {
        // a new bandit, and a new agent, every run
        game_type game(10, static_cast<unsigned int>(run));
        std::mt19937 prng(static_cast<unsigned int>(run) + 1000);
        types::policy memory;
        relearn::q_probabilistic<types::state, types::action> learner(1.0);
        train(game, learner, memory, 1000, 0.1, 1, prng);
        // the arm the agent now believes is best
        auto arm = games::greedy<game_type>(memory, 0, game.actions(0), prng);
        found  += arm == game.best_arm();
        regret += game.mean(game.best_arm()) - game.mean(arm);
    }
    REQUIRE(found >= 0.75 * runs);
    REQUIRE(regret / runs < 0.06);
}

// ---------------------------------------------------------------- random walk

/// verifies the rules of the random walk and its closed-form values (Sutton & Barto, Example 6.2)
TEST_CASE("random walk: rules", "[games][random_walk]")
{
    games::random_walk walk(5, 0, 1, 2);
    int right = 0;
    for (int i = 0; i < 2000; i++) {
        // every walk starts in the middle, then steps left or right at random
        auto s = walk.reset();
        REQUIRE(s == 3u);
        auto out = walk.step(0);
        REQUIRE((out.next == 2u || out.next == 4u));
        right += out.next == 4u;
    }
    REQUIRE(right == Approx(1000).epsilon(0.1));
    REQUIRE(walk.true_value(1) == Approx(1.0 / 6));
    REQUIRE(walk.true_value(5) == Approx(5.0 / 6));
    games::random_walk long_walk(19, -1, 1);
    REQUIRE(long_walk.true_value(10) == Approx(0.0));
}

/// verifies that `q_probabilistic` predicts the true values of the 5 and 19 state walks
TEST_CASE("random walk: q_probabilistic predicts the true values", "[games][random_walk]")
{
    using game_type = games::random_walk;
    using types     = games::types<game_type>;
    for (unsigned int n : {5u, 19u}) {
        // the 5 state walk ends in 0 or +1, the 19 state walk in -1 or +1
        game_type walk(n, n == 5 ? 0 : -1, 1, 1);
        std::mt19937 prng(2);
        types::policy memory;
        relearn::q_probabilistic<types::state, types::action> learner(1.0);
        train(walk, learner, memory, 10000, 0, 100000, prng);
        for (unsigned int s = 1; s <= n; s++) {
            INFO("states " << n << " state " << s);
            REQUIRE(memory.value(types::state(s), types::action(0)) ==
                    Approx(walk.true_value(s)).margin(n == 5 ? 0.02 : 0.05));
        }
    }
}

// ------------------------------------------------------------- short corridor

/// verifies the rules of the short corridor with switched actions (Sutton & Barto, Example 13.1)
TEST_CASE("short corridor: rules", "[games][short_corridor]")
{
    using game_type = games::short_corridor;
    game_type game;
    REQUIRE(game.reset() == 0u);
    REQUIRE(game.cell() == 0u);
    REQUIRE(game.actions(0).size() == 2);
    // the first cell: left bumps into the wall, right moves on. Every step is worth -1,
    // and the agent always sees 0
    auto out = game.step(game_type::go_left);
    REQUIRE(game.cell() == 0u);
    REQUIRE(out.next == 0u);
    REQUIRE(out.reward == -1);
    REQUIRE_FALSE(out.terminated);
    out = game.step(game_type::go_right);
    REQUIRE(game.cell() == 1u);
    REQUIRE(out.next == 0u);
    // the second cell switches the actions: right goes back, left goes on
    game.step(game_type::go_right);
    REQUIRE(game.cell() == 0u);
    game.step(game_type::go_right);
    out = game.step(game_type::go_left);
    REQUIRE(game.cell() == 2u);
    REQUIRE(out.next == 0u);
    // the third cell: left goes back, right reaches G
    game.step(game_type::go_left);
    REQUIRE(game.cell() == 1u);
    game.step(game_type::go_left);
    out = game.step(game_type::go_right);
    REQUIRE(game.cell() == 3u);
    REQUIRE(out.reward == -1);
    REQUIRE(out.terminated);
    // with the cells shown, the agent sees which one it is in
    game_type shown;
    shown.show_position(true);
    REQUIRE(shown.reset() == 0u);
    REQUIRE(shown.step(game_type::go_right).next == 1u);
    REQUIRE(shown.step(game_type::go_left).next == 2u);
    REQUIRE(shown.step(game_type::go_right).terminated);
}

/// verifies the exact value of the start, against the iterative evaluation of `solvers::tabular`
TEST_CASE("short corridor: the exact values", "[games][short_corridor]")
{
    using game_type = games::short_corridor;
    // the corridor as a tabular model: cells 0, 1, 2, actions left (0) and right (1)
    solvers::tabular corridor;
    corridor.states  = 3;
    corridor.actions = 2;
    corridor.gamma   = 1;
    corridor.model = [](int s, int a) {
        // the second cell switches the actions round, and left in the first cell bumps into the wall
        bool goes_right = (a == 1) != (s == 1);
        int next = goes_right ? s + 1 : std::max(s - 1, 0);
        return std::vector<solvers::outcome> {{1.0, next == 3 ? 0 : next, -1.0, next == 3}};
    };
    // the same probability of right in every cell, and three different ones
    const std::vector<std::vector<double>> rights = {{0.5, 0.5, 0.5}, {0.05, 0.05, 0.05},
                                                     {0.95, 0.95, 0.95}, {0.9, 0.2, 0.7}};
    for (const auto & p : rights) {
        std::vector<std::vector<double>> pi;
        for (double right : p) {
            pi.push_back({1 - right, right});
        }
        double exact = corridor.policy_values(pi)[0];
        REQUIRE(game_type::start_value(p[0], p[1], p[2]) == Approx(exact));
        if (p[0] == p[1]) {
            REQUIRE(game_type::start_value(p[0]) == Approx(exact));
        }
    }
    // Example 13.1: ε-greedy (ε = 0.1) right is worth less than -44, left less than -82
    REQUIRE(game_type::start_value(0.95) == Approx(-44.21).margin(0.01));
    REQUIRE(game_type::start_value(0.05) == Approx(-82.11).margin(0.01));
    // the best probability of right, 2 - √2, is worth -(6 + 4√2) = -11.657, and is the best
    const double best = game_type::best_right();
    REQUIRE(best == Approx(0.5858).margin(0.0001));
    REQUIRE(game_type::start_value(best) == Approx(-(6 + 4 * std::sqrt(2.0))));
    REQUIRE(game_type::start_value(best) > game_type::start_value(best - 0.01));
    REQUIRE(game_type::start_value(best) > game_type::start_value(best + 0.01));
    // and the game agrees: going right half the time takes 12 steps on average
    game_type game;
    std::mt19937 prng(1);
    std::bernoulli_distribution coin(0.5);
    auto agent = [&](const unsigned int &,
                     const std::vector<unsigned int> &) {
        return coin(prng) ? game_type::go_right : game_type::go_left;
    };
    double total = 0;
    for (int i = 0; i < 20000; i++) {
        total += games::play(game, agent, 100000).total_reward;
    }
    REQUIRE(total / 20000 == Approx(game_type::start_value(0.5)).margin(0.3));
}

// ---------------------------------------------------------- maximization bias

/// verifies the rules of the maximization bias game (Sutton & Barto, Example 6.7)
TEST_CASE("maximization bias: rules", "[games][max_bias]")
{
    using game_type = games::maximization_bias;
    game_type game(10, 1);
    REQUIRE(game.reset() == game_type::A);
    // right ends the game at once, worth 0
    auto out = game.step(game_type::right);
    REQUIRE(out.terminated);
    REQUIRE(out.reward == 0);
    // left leads to B, whose 10 actions are worth -0.1 on average
    double sum = 0;
    const int n = 20000;
    for (int i = 0; i < n; i++) {
        game.reset();
        out = game.step(game_type::left);
        REQUIRE(out.next == game_type::B);
        REQUIRE(!out.terminated);
        REQUIRE(game.actions(out.next).size() == 10u);
        out = game.step(static_cast<unsigned int>(i % 10));
        REQUIRE(out.terminated);
        sum += out.reward;
    }
    REQUIRE(sum / n == Approx(-0.1).margin(0.03));
}

/// verifies that Q-learning shows the maximization bias of Sutton & Barto, Figure 6.5
TEST_CASE("maximization bias: q_learning goes left far more often than it should", "[games][max_bias]")
{
    // going left is worse on average, so the best epsilon-greedy agent goes left
    // 5% of the time (epsilon / 2); Q-learning's max over B's noisy values makes
    // it go left much more often during its first episodes (Sutton & Barto, Figure 6.5)
    using game_type = games::maximization_bias;
    using types     = games::types<game_type>;
    const int runs = 1000;
    const int episodes = 100;
    double left = 0;
    for (int run = 0; run < runs; run++) {
        game_type game(10, static_cast<unsigned int>(run));
        std::mt19937 prng(static_cast<unsigned int>(run) + 5000);
        types::policy memory;
        relearn::q_learning<types::state, types::action> learner;
        learner.alpha = 0.1;
        learner.gamma = 1.0;
        // the ε-greedy agent
        auto agent = [&](const unsigned int & s,
                         const std::vector<unsigned int> & legal) {
            return games::epsilon_greedy<game_type>(memory, s, legal, 0.1, prng);
        };
        for (int i = 0; i < episodes; i++) {
            auto r = games::play(game, agent, 10);
            // an episode of two steps went left (through B)
            left += r.steps == 2;
            learner(r.episode, memory, r.terminated);
        }
    }
    REQUIRE(left / (runs * episodes) > 0.25);
}

// ----------------------------------------------------------- gambler's problem

namespace
{
/**
 * @brief the exact model of the gambler's problem, written from Sutton & Barto's rules
 * @param p_heads is the probability that the coin comes up heads
 * @return the model: the state is the capital (0 to 100), the action the stake
 */
solvers::tabular gambler_model(double p_heads)
{
    solvers::tabular m;
    m.states  = 101;
    m.actions = 51;
    m.gamma   = 1;
    m.model = [p_heads](int s, int stake) {
        std::vector<solvers::outcome> out;
        // the game is over at 0 and 100, and a stake must be between 1 and min(s, 100 - s)
        if (s <= 0 || s >= 100 || stake < 1 || stake > std::min(s, 100 - s)) {
            return out;
        }
        // heads wins the stake (reaching 100 is worth 1), tails loses it
        out.push_back({p_heads, s + stake, s + stake >= 100 ? 1.0 : 0.0, s + stake >= 100});
        out.push_back({1 - p_heads, s - stake, 0.0, s - stake == 0});
        return out;
    };
    return m;
}
} // namespace

/// verifies the exact values of the gambler's problem (Sutton & Barto, Example 4.3) and its rules
TEST_CASE("gambler's problem: the exact values", "[games][gambler]")
{
    // with p = 0.4: V(25) = p^2, V(50) = p, V(75) = p + (1 - p) p
    auto v = gambler_model(0.4).optimal_values();
    REQUIRE(v[25] == Approx(0.16));
    REQUIRE(v[50] == Approx(0.4));
    REQUIRE(v[75] == Approx(0.64));
    games::gamblers_problem game(0.4, 50, 100, 1);
    REQUIRE(game.reset() == 50u);
    REQUIRE(game.actions(50).size() == 50u);
    REQUIRE(game.actions(99).size() == 1u);
    int wins = 0;
    for (int i = 0; i < 10000; i++) {
        game.reset();
        // all in: won with probability p
        auto out = game.step(50);
        REQUIRE(out.terminated);
        wins += out.reward == 1;
    }
    REQUIRE(wins == Approx(4000).epsilon(0.05));
}

/// verifies, by exact evaluation, that the learnt stakes are nearly optimal
TEST_CASE("gambler's problem: q_probabilistic learns near-optimal stakes", "[games][gambler]")
{
    using game_type = games::gamblers_problem;
    using types     = games::types<game_type>;
    // random starting capital: every state is visited
    game_type game(0.4, 0, 100, 1);
    std::mt19937 prng(3);
    types::policy memory;
    relearn::q_probabilistic<types::state, types::action> learner(1.0);
    train(game, learner, memory, 100000, 0.1, 1000, prng);
    // evaluate the learnt greedy stakes exactly, and compare with the optimal values
    auto exact = gambler_model(0.4);
    auto best = exact.optimal_values();
    std::vector<int> policy(101, 1);
    for (unsigned int s = 1; s < 100; s++) {
        policy[s] = static_cast<int>(games::greedy<game_type>(memory, s, game.actions(s), prng));
    }
    auto learnt = exact.policy_values(policy);
    double mean_gap  = 0;
    double worst_gap = 0;
    for (int s = 1; s < 100; s++) {
        mean_gap += (best[s] - learnt[s]) / 99;
        worst_gap = std::max(worst_gap, best[s] - learnt[s]);
    }
    REQUIRE(mean_gap < 0.02);
    REQUIRE(worst_gap < 0.05);
}

// ------------------------------------------------------------------ dyna maze

namespace
{
/**
 * @brief the length of the shortest path from the start to the goal, by breadth-first search
 * @param maze is the maze, with its current walls
 * @return the number of moves, or -1 if the goal cannot be reached
 */
int shortest_path(const games::dyna_maze & maze)
{
    std::map<std::pair<int, int>, int> distance;
    std::queue<games::position> frontier;
    frontier.push(maze.start());
    distance[{maze.start().x, maze.start().y}] = 0;
    while (!frontier.empty()) {
        auto p = frontier.front();
        frontier.pop();
        // cells leave the queue in order of distance: the first time we see the goal is the shortest
        if (p == games::dyna_maze::goal()) {
            return distance[{p.x, p.y}];
        }
        for (unsigned int a = 0; a < 4; a++) {
            // a wall keeps the agent in place
            auto n = games::moved(p, a, games::dyna_maze::width, games::dyna_maze::height);
            if (maze.wall(n)) {
                n = p;
            }
            if (!distance.count({n.x, n.y})) {
                distance[{n.x, n.y}] = distance[{p.x, p.y}] + 1;
                frontier.push(n);
            }
        }
    }
    return -1;
}
} // namespace

/// verifies the shortest paths of the mazes of Sutton & Barto, Figures 8.2, 8.4 and 8.5
TEST_CASE("dyna maze: layouts", "[games][dyna_maze]")
{
    using maze = games::dyna_maze;
    // the shortest path through the maze of Figure 8.2 takes 14 moves
    REQUIRE(shortest_path(maze(maze::maze)) == 14);
    // blocking: the short way (right) closes, the new way (left) is longer
    maze blocking(maze::blocking, 5);
    REQUIRE(!blocking.changed());
    REQUIRE(shortest_path(blocking) == 10);
    blocking.reset();
    for (int i = 0; i < 5; i++) {
        blocking.step(games::left);
    }
    REQUIRE(blocking.changed());
    REQUIRE(shortest_path(blocking) == 16);
    // shortcut: a shorter way opens on the right
    maze shortcut(maze::shortcut, 0);
    REQUIRE(shortcut.changed());
    REQUIRE(shortest_path(shortcut) == 10);
    REQUIRE(shortest_path(maze(maze::shortcut, 1000)) == 16);
    // walls keep the agent in place; the goal is worth +1 and ends the episode
    maze m;
    m.place({1, 2});
    REQUIRE(m.step(games::right).next == (games::position{1, 2}));
    m.place({8, 1});
    auto out = m.step(games::up);
    REQUIRE(out.terminated);
    REQUIRE(out.reward == 1);
}

/// verifies that Q-learning learns a route through the maze of Sutton & Barto, Figure 8.2
TEST_CASE("dyna maze: q_learning finds one of the two shortest routes", "[games][dyna_maze]")
{
    // the maze has a 14-move and a 16-move route; telling them apart takes plain
    // Q-learning many episodes of exploration (Dyna-Q's planning does it faster)
    using game_type = games::dyna_maze;
    using types     = games::types<game_type>;
    game_type maze;
    std::mt19937 prng(1);
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.5;
    learner.gamma = 0.95;
    train(maze, learner, memory, 1000, 0.1, 10000, prng);
    // play once greedily
    auto agent = [&](const game_type::state_type & s,
                     const std::vector<unsigned int> & legal) {
        return games::greedy<game_type>(memory, s, legal, prng);
    };
    auto greedy = games::play(maze, agent, 100);
    REQUIRE(greedy.terminated);
    REQUIRE(greedy.steps <= 16u);
}

// ---------------------------------------------------------------- tic-tac-toe

/// verifies the board's winners, minimax's draw, and that both opponents play legal moves
TEST_CASE("tic-tac-toe: rules", "[games][tic_tac_toe]")
{
    games::board b;
    REQUIRE(b.winner() == 0);
    REQUIRE(b.empty().size() == 9u);
    b.cells = {1, 1, 1, 2, 2, 0, 0, 0, 0};
    REQUIRE(b.winner() == 1);
    // a diagonal
    b.cells = {2, 1, 1, 1, 2, 0, 1, 0, 2};
    REQUIRE(b.winner() == 2);
    b.cells = {1, 2, 1, 1, 2, 2, 2, 1, 1};
    REQUIRE(b.winner() == 0);
    REQUIRE(b.full());

    // perfect play is a draw
    games::tic_tac_toe perfect(games::tic_tac_toe::perfect);
    REQUIRE(perfect.minimax(games::board(), 1) == 0);

    // a random X never beats the perfect O, and the opponent only plays empty cells
    games::tic_tac_toe random_o(games::tic_tac_toe::random, 3);
    std::mt19937 prng(4);
    for (int i = 0; i < 500; i++) {
        for (auto * game : {&perfect, &random_o}) {
            auto s = game->reset();
            for (;;) {
                // X plays a random empty cell
                auto moves = game->actions(s);
                std::uniform_int_distribution<std::size_t> any(0, moves.size() - 1);
                auto before = s;
                auto out = game->step(moves[any(prng)]);
                // no cell taken before changes, and one or two empty cells are taken (X, then O)
                int placed = 0;
                for (int c = 0; c < 9; c++) {
                    if (before.cells[c] != 0) {
                        REQUIRE(out.next.cells[c] == before.cells[c]);
                    }
                    placed += before.cells[c] == 0 && out.next.cells[c] != 0;
                }
                REQUIRE((placed == 1 || placed == 2));
                if (out.terminated) {
                    if (game == &perfect) {
                        REQUIRE(out.reward <= 0);
                    }
                    break;
                }
                s = out.next;
            }
        }
    }
}

/// verifies that Q-learning beats a random opponent, and learns to draw a perfect (minimax) one
TEST_CASE("tic-tac-toe: q_learning beats a random opponent and draws a perfect one", "[games][tic_tac_toe]")
{
    using game_type = games::tic_tac_toe;
    using types     = games::types<game_type>;
    // play greedily, @return the counts of wins, draws and losses
    auto results = [](game_type & game,
                      const types::policy & memory,
                      int games_played,
                      std::mt19937 & prng) {
        std::vector<int> wdl(3, 0);
        auto agent = [&](const game_type::state_type & s,
                         const std::vector<unsigned int> & legal) {
            return games::greedy<game_type>(memory, s, legal, prng);
        };
        for (int i = 0; i < games_played; i++) {
            auto r = games::play(game, agent, 9);
            wdl[r.total_reward > 0 ? 0 : (r.total_reward == 0 ? 1 : 2)]++;
        }
        return wdl;
    };
    std::mt19937 prng(2);
    {
        // trained and tested against a random opponent
        game_type game(game_type::random, 1);
        types::policy memory;
        relearn::q_learning<types::state, types::action> learner;
        learner.alpha = 0.3;
        learner.gamma = 1.0;
        train(game, learner, memory, 20000, 0.1, 9, prng);
        game_type test(game_type::random, 7);
        auto wdl = results(test, memory, 10000, prng);
        // wins
        REQUIRE(wdl[0] >= 9000);
        // losses
        REQUIRE(wdl[2] <= 200);
    }
    {
        // trained and tested against the perfect opponent, which is deterministic
        game_type game(game_type::perfect);
        types::policy memory;
        relearn::q_learning<types::state, types::action> learner;
        learner.alpha = 0.3;
        learner.gamma = 1.0;
        train(game, learner, memory, 20000, 0.1, 9, prng);
        auto wdl = results(game, memory, 1, prng);
        // a draw: the best possible result
        REQUIRE(wdl[1] == 1);
    }
}

// ------------------------------------------------------------------ cart-pole

/// verifies the cart-pole physics against a step worked out by hand from Gymnasium's equations
TEST_CASE("cart-pole: physics", "[games][cart_pole]")
{
    games::cart_pole game(1);
    // every episode starts near rest
    for (int i = 0; i < 100; i++) {
        for (double v : game.reset()) {
            REQUIRE(v >= -0.05);
            REQUIRE(v <= 0.05);
        }
    }
    // one push to the right from rest, worked out by hand from the equations:
    // temp = 10 / 1.1, angular acc. = -temp / (0.5 (4/3 - 0.1 / 1.1)) = -14.634146,
    // cart acc. = temp + 0.05 * 14.634146 / 1.1 = 9.756098, then one Euler step of 0.02 s
    game.place({0, 0, 0, 0});
    auto out = game.step(games::cart_pole::push_right);
    REQUIRE(out.next[0] == Approx(0.0));
    REQUIRE(out.next[1] == Approx(0.195122).epsilon(1e-5));
    REQUIRE(out.next[2] == Approx(0.0));
    REQUIRE(out.next[3] == Approx(-0.292683).epsilon(1e-5));
    REQUIRE(out.reward == 1);
    REQUIRE(!out.terminated);
    // pushing left is the mirror image
    game.place({0, 0, 0, 0});
    out = game.step(games::cart_pole::push_left);
    REQUIRE(out.next[1] == Approx(-0.195122).epsilon(1e-5));
    REQUIRE(out.next[3] == Approx(0.292683).epsilon(1e-5));
    // leaning past 12 degrees, or leaving the track, ends the episode (still worth +1)
    const double angle = games::cart_pole::angle_limit;
    game.place({0, 0, angle - 0.001, 1.0});
    out = game.step(games::cart_pole::push_right);
    REQUIRE(out.terminated);
    REQUIRE(out.reward == 1);
    game.place({2.39, 1.0, 0, 0});
    REQUIRE(game.step(games::cart_pole::push_right).terminated);
}

/// verifies that Q-learning on 162 boxes balances the pole for longer as it learns
TEST_CASE("cart-pole: q_learning on a discretised state learns to balance longer", "[games][cart_pole]")
{
    using game_type = games::discretised<games::cart_pole>;
    using types     = games::types<game_type>;
    // boxes after Barto, Sutton & Anderson: position, velocity, angle, angular velocity
    game_type game(games::cart_pole(1), {{-0.8, 0.8, 3},
                                         {-0.5, 0.5, 3},
                                         {-0.21, 0.21, 6},
                                         {-0.87, 0.87, 3}});
    REQUIRE(game.cells() == 162u);
    std::mt19937 prng(1);
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.1;
    learner.gamma = 0.99;
    // the ε-greedy agent
    auto agent = [&](const unsigned int & s,
                     const std::vector<unsigned int> & legal) {
        return games::epsilon_greedy<game_type>(memory, s, legal, 0.05, prng);
    };
    double first = 0;
    double last  = 0;
    for (int i = 0; i < 10000; i++) {
        auto r = games::play(game, agent, game.inner().step_limit());
        learner(r.episode, memory, r.terminated);
        // the average length of the first and of the last 1000 episodes
        if (i < 1000) {
            first += r.steps / 1000.0;
        }
        if (i >= 9000) {
            last += r.steps / 1000.0;
        }
    }
    INFO("average steps: first 1000 episodes " << first << ", last 1000 " << last);
    // across seeds 1 ... 10 the first 1000 episodes averaged 17 ... 110 steps,
    // and the last 1000 averaged 232 ... 339, at least 2.7 times as many
    REQUIRE(last > 2 * first);
    REQUIRE(last > 150);
}

// --------------------------------------------------------------- mountain car

/// verifies the mountain car physics, its walls and speed limit, and that rocking reaches the flag
TEST_CASE("mountain car: physics", "[games][mountain_car]")
{
    games::mountain_car car(1);
    // every episode starts at rest, between -0.6 and -0.4
    for (int i = 0; i < 100; i++) {
        auto s = car.reset();
        REQUIRE(s[0] >= -0.6);
        REQUIRE(s[0] <= -0.4);
        REQUIRE(s[1] == 0);
    }
    // coasting from rest at -0.5: velocity = cos(-1.5) * -0.0025
    car.place({-0.5, 0});
    auto out = car.step(games::mountain_car::coast);
    REQUIRE(out.next[1] == Approx(std::cos(-1.5) * -0.0025));
    REQUIRE(out.next[0] == Approx(-0.5 + std::cos(-1.5) * -0.0025));
    REQUIRE(out.reward == -1);
    // the car stops against the left wall
    car.place({-1.19, -0.05});
    out = car.step(games::mountain_car::accelerate_left);
    REQUIRE(out.next[0] == Approx(-1.2));
    REQUIRE(out.next[1] == 0);
    // the speed is limited to 0.07 (at -0.8 gravity and engine both speed the car up)
    car.place({-0.8, 0.07});
    REQUIRE(car.step(games::mountain_car::accelerate_right).next[1] == Approx(0.07));
    // reaching 0.5 ends the episode
    car.place({0.49, 0.05});
    REQUIRE(car.step(games::mountain_car::coast).terminated);
    // the engine alone cannot climb the hill, but rocking back and forth
    // (accelerating in the direction of motion) reaches the flag in time
    for (int i = 0; i < 100; i++) {
        auto s = car.reset();
        bool reached = false;
        for (std::size_t t = 0; t < car.step_limit() && !reached; t++) {
            auto o = car.step(s[1] < 0 ? games::mountain_car::accelerate_left
                                       : games::mountain_car::accelerate_right);
            s = o.next;
            reached = o.terminated;
        }
        REQUIRE(reached);
    }
}

/// verifies that Q-learning on a 20 x 20 grid of the state learns to reach the flag
TEST_CASE("mountain car: q_learning on a discretised state learns to reach the flag", "[games][mountain_car]")
{
    using game_type = games::discretised<games::mountain_car>;
    using types     = games::types<game_type>;
    game_type game(games::mountain_car(1), {{-1.2, 0.6, 20},
                                            {-0.07, 0.07, 20}});
    std::mt19937 prng(1);
    types::policy memory;
    relearn::q_learning<types::state, types::action> learner;
    learner.alpha = 0.1;
    learner.gamma = 1.0;
    // greedy: values start at 0 and every step costs -1, so untried moves look
    // best and the agent explores without random moves (optimistic start)
    auto agent = [&](const unsigned int & s,
                     const std::vector<unsigned int> & legal) {
        return games::greedy<game_type>(memory, s, legal, prng);
    };
    double last = 0;
    for (int i = 0; i < 5000; i++) {
        auto r = games::play(game, agent, game.inner().step_limit());
        learner(r.episode, memory, r.terminated);
        // the average return of the last 500 episodes
        if (i >= 4500) {
            last += r.total_reward / 500;
        }
    }
    INFO("average return of the last 500 episodes " << last);
    // -200 means the flag is never reached
    REQUIRE(last > -175);
}

/// verifies how `games::bins` and `games::discretised` cut a continuous state into cells
TEST_CASE("discretised: bins and cells", "[games][discretised]")
{
    // 4 bins over [-1, 1]; values outside fall into the first or last bin
    games::bins b{-1.0, 1.0, 4};
    REQUIRE(b.of(-5) == 0u);
    REQUIRE(b.of(-0.9) == 0u);
    REQUIRE(b.of(-0.4) == 1u);
    REQUIRE(b.of(0.1) == 2u);
    REQUIRE(b.of(0.99) == 3u);
    REQUIRE(b.of(7) == 3u);
    // 3 x 2 cells, numbered with the last variable changing fastest
    games::discretised<games::mountain_car> game(games::mountain_car(), {{-1.2, 0.6, 3},
                                                                         {-0.07, 0.07, 2}});
    REQUIRE(game.cells() == 6u);
    REQUIRE(game.cell({-1.2, -0.07}) == 0u);
    REQUIRE(game.cell({-1.2, 0.07}) == 1u);
    REQUIRE(game.cell({0.6, 0.07}) == 5u);
}

// ------------------------------------------------------------ games::play

namespace
{
/**
 * @struct endless_corridor
 * @brief a corridor that never ends: each step moves one cell on and is worth 2
 */
struct endless_corridor
{
    using state_type  = unsigned int;
    using action_type = unsigned int;
    /// the cell the agent is in
    state_type at = 0;

    /// @brief start a new episode @return the first cell
    state_type reset()
    {
        at = 0;
        return at;
    }

    /// @brief move one cell on @return the next cell, and a reward of 2
    games::step_result<state_type> step(const action_type &)
    {
        games::step_result<state_type> out;
        out.next = ++at;
        out.reward = 2;
        return out;
    }

    /// @return the only action: go on
    std::vector<action_type> actions(const state_type &) const
    {
        return {0};
    }
};
} // namespace

/// verifies that `games::play` records a cut-short episode so that its last step is still learnt
TEST_CASE("play: an episode cut short still learns its last step", "[games][play]")
{
    // the last step taken before the cut-off is learnt from its reward and the
    // value of the state it reached: one step, cut short, Q(0) = 2 + 0.5 * Q(1)
    using types = games::types<endless_corridor>;
    endless_corridor game;
    auto always_on = [](const unsigned int &,
                        const std::vector<unsigned int> &) {
        return 0u;
    };
    auto r = games::play(game, always_on, 1);
    REQUIRE(!r.terminated);
    REQUIRE(r.steps == 1u);
    REQUIRE(r.total_reward == 2);

    SECTION("q_learning") {
        types::policy memory;
        // what state 1 is worth
        memory.update(types::state(1), types::action(0), 3.0);
        relearn::q_learning<types::state, types::action> learner;
        learner.alpha = 1;
        learner.gamma = 0.5;
        learner(r.episode, memory, r.terminated);
        REQUIRE(memory.value(types::state(0), types::action(0)) == Approx(3.5));
        // the last link only holds the state reached: it is not learnt
        REQUIRE(memory.value(types::state(1), types::action(0)) == Approx(3.0));
    }
    SECTION("q_probabilistic") {
        types::policy memory;
        memory.update(types::state(1), types::action(0), 3.0);
        relearn::q_probabilistic<types::state, types::action> learner(0.5);
        learner(r.episode, memory, r.terminated);
        REQUIRE(memory.value(types::state(0), types::action(0)) == Approx(3.5));
        REQUIRE(memory.value(types::state(1), types::action(0)) == Approx(3.0));
    }
}
