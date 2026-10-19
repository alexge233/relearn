#ifndef RELEARN_GAMES_MOUNTAIN_CAR_HPP
#define RELEARN_GAMES_MOUNTAIN_CAR_HPP
/**
 * Mountain car (Moore, 1990; Sutton & Barto, Example 10.1), with the physics,
 * constants and limits of Gymnasium's `MountainCar-v0`.
 *
 * - An underpowered car sits in a valley and must reach the flag on the right
 *   hill (position 0.5). Its engine is too weak to climb directly: it has to
 *   rock back and forth to build up speed.
 * - The state is (position in [-1.2, 0.6], velocity in [-0.07, 0.07]); each
 *   episode starts at rest at a position drawn uniformly from [-0.6, -0.4].
 * - Actions: 0 accelerate left, 1 do nothing, 2 accelerate right.
 * - Every step is worth -1. The episode ends (terminated) at the flag;
 *   Gymnasium stops it (truncated) after 200 steps: use `step_limit()`.
 * - The state is continuous: wrap the game in `games::discretised` to learn it
 *   with a tabular learner.
 */
#include "game.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace games
{
class mountain_car
{
public:
    using state_type  = std::array<double, 2>;    // position, velocity
    using action_type = unsigned int;
    enum : action_type { accelerate_left = 0, coast = 1, accelerate_right = 2 };

    static constexpr double min_position = -1.2;
    static constexpr double max_position = 0.6;
    static constexpr double max_speed = 0.07;
    static constexpr double goal_position = 0.5;
    static constexpr double force = 0.001;
    static constexpr double gravity = 0.0025;

    explicit mountain_car(unsigned int seed = 0)
    : prng_(seed)
    {}

    std::size_t step_limit() const { return 200; }

    state_type reset()
    {
        std::uniform_real_distribution<double> start(-0.6, -0.4);
        s_ = {start(prng_), 0.0};
        return s_;
    }

    /// set the state to @param s (for tests)
    void place(const state_type & s) { s_ = s; }

    step_result<state_type> step(const action_type & a)
    {
        // local copies: std::min/max take references, which C++14 static
        // constexpr members cannot bind to without a separate definition
        const double speed_limit = max_speed, left_end = min_position, right_end = max_position;
        double position = s_[0], velocity = s_[1];
        velocity += (static_cast<double>(a) - 1.0) * force + std::cos(3 * position) * (-gravity);
        velocity = std::max(-speed_limit, std::min(speed_limit, velocity));
        position += velocity;
        position = std::max(left_end, std::min(right_end, position));
        if (position == left_end && velocity < 0) {
            velocity = 0;                       // the car stops against the left wall
        }
        s_ = {position, velocity};

        step_result<state_type> out;
        out.next = s_;
        out.reward = -1;
        out.terminated = position >= goal_position;
        return out;
    }

    std::vector<action_type> actions(const state_type &) const
    {
        return {accelerate_left, coast, accelerate_right};
    }

private:
    std::mt19937 prng_;
    state_type s_{};
};
} // namespace games
#endif
