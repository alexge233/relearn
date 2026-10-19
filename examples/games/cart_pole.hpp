#ifndef RELEARN_GAMES_CART_POLE_HPP
#define RELEARN_GAMES_CART_POLE_HPP
/**
 * Cart-pole balancing (Barto, Sutton & Anderson, 1983), with the physics,
 * constants and limits of Gymnasium's `CartPole-v1`.
 *
 * - A pole is hinged on a cart that moves along a track. Each step pushes the
 *   cart left (action 0) or right (action 1) with a force of 10 N, for 0.02 s.
 * - The state is (cart position, cart velocity, pole angle, pole angular velocity),
 *   each starting uniformly in [-0.05, 0.05].
 * - Every step is worth +1, including the last. The episode ends (terminated)
 *   when the pole leans more than 12 degrees or the cart leaves [-2.4, 2.4];
 *   Gymnasium stops it (truncated) after 500 steps: use `step_limit()`.
 * - The state is continuous: wrap the game in `games::discretised` to learn it
 *   with a tabular learner.
 */
#include "game.hpp"
#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace games
{
/**
 * @class cart_pole
 * @brief balance a pole on a cart by pushing the cart left or right
 */
class cart_pole
{
public:
    /// a state is the cart position, cart velocity, pole angle and pole angular velocity
    using state_type  = std::array<double, 4>;
    /// an action is a push to the left or to the right
    using action_type = unsigned int;
    /// the two pushes
    enum : action_type
    {
        push_left  = 0,
        push_right = 1
    };

    // the physical constants of Gymnasium's CartPole-v1
    static constexpr double gravity          = 9.8;
    static constexpr double cart_mass        = 1.0;
    static constexpr double pole_mass        = 0.1;
    static constexpr double pole_half_length = 0.5;
    static constexpr double force            = 10.0;
    // seconds per step
    static constexpr double tau              = 0.02;
    // the episode ends past 12 degrees, or when the cart leaves [-2.4, 2.4]
    static constexpr double angle_limit      = 12 * 2 * 3.14159265358979323846 / 360;
    static constexpr double position_limit   = 2.4;

    /// @brief create the game @param seed seeds the random starting state
    explicit cart_pole(unsigned int seed = 0)
    : prng_(seed)
    {}

    /// @return Gymnasium's step limit, after which an episode is cut short
    std::size_t step_limit() const
    {
        return 500;
    }

    /// @brief start a new episode @return a state near upright and at rest
    state_type reset()
    {
        std::uniform_real_distribution<double> start(-0.05, 0.05);
        for (auto & v : s_) {
            v = start(prng_);
        }
        return s_;
    }

    /// @brief set the state to @param s (for tests)
    void place(const state_type & s)
    {
        s_ = s;
    }

    /// @brief push the cart with action @param a for one step of 0.02 s
    /// @return the state reached, the reward and whether the pole fell or the cart left the track
    step_result<state_type> step(const action_type & a)
    {
        const double total_mass = cart_mass + pole_mass;
        const double pole_mass_length = pole_mass * pole_half_length;
        // the force of the push: to the right is positive
        double push = a == push_right ? force : -force;
        double x         = s_[0];
        double x_dot     = s_[1];
        double theta     = s_[2];
        double theta_dot = s_[3];
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);
        // the equations of motion, as in Gymnasium (after Florian, 2007)
        double temp = (push + pole_mass_length * theta_dot * theta_dot * sin_t) / total_mass;
        double theta_acc = (gravity * sin_t - cos_t * temp) /
                           (pole_half_length * (4.0 / 3.0 - pole_mass * cos_t * cos_t / total_mass));
        double x_acc = temp - pole_mass_length * theta_acc * cos_t / total_mass;
        // Euler integration, as Gymnasium's default
        x         += tau * x_dot;
        x_dot     += tau * x_acc;
        theta     += tau * theta_dot;
        theta_dot += tau * theta_acc;
        s_ = {x, x_dot, theta, theta_dot};

        // every step is worth +1; the episode ends when the pole falls or the cart leaves the track
        step_result<state_type> out;
        out.next = s_;
        out.reward = 1;
        out.terminated = x < -position_limit || x > position_limit ||
                         theta < -angle_limit || theta > angle_limit;
        return out;
    }

    /// @return the legal actions: push left or push right, in every state
    std::vector<action_type> actions(const state_type &) const
    {
        return {push_left, push_right};
    }

private:
    // the random starting state
    std::mt19937 prng_;
    // the current state
    state_type s_{};
};
} // namespace games
#endif
