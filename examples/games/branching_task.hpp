#ifndef RELEARN_GAMES_BRANCHING_TASK_HPP
#define RELEARN_GAMES_BRANCHING_TASK_HPP
/**
 * The random tasks of Sutton & Barto, Section 8.6 (Figure 8.8): many states, two actions in each, and a few
 * branches for every action, drawn at random.
 *
 * - States 0 ... n-1, and n, the terminal state; every episode starts in state 0.
 * - Two actions in every state. Each state/action pair leads to one of `b` next states (its *branches*), drawn
 *   uniformly from all the states when the task is made (each one on its own, so two of them can be the same state),
 *   each with probability 0.9 / b; and with probability 0.1
 *   to the terminal state, so the episodes end after 10 steps on average.
 * - Each of those transitions has its own reward, drawn from a normal distribution of mean 0 and variance 1 when
 *   the task is made (the book gives the transitions an expected reward drawn so).
 *
 * It is a model only, for planning (`model()`): nothing plays it.
 */
#include "game.hpp"
#include <random>
#include <vector>

namespace games
{
/**
 * @class branching_task
 * @brief a random task of Section 8.6: n states, 2 actions, b branches for each pair, and a 0.1 chance of ending
 */
class branching_task
{
public:
    /// a state is its number, n being the terminal state
    using state_type  = unsigned int;
    /// an action is 0 or 1
    using action_type = unsigned int;
    /// the rules as a model, for planning
    using model_type  = relearn::finite_mdp<relearn::state<state_type>,
                                            relearn::action<action_type>>;

    /**
     * @brief a new random task
     * @param states is n, the number of states (but the terminal one)
     * @param branches is b, the number of next states of each pair
     * @param seed draws the branches and the rewards
     */
    branching_task(unsigned int states,
                   unsigned int branches,
                   unsigned int seed)
    : n_(states),
      b_(branches)
    {
        std::mt19937 prng(seed);
        std::uniform_int_distribution<unsigned int> pick(0, n_ - 1);
        std::normal_distribution<double> gauss(0, 1);
        next_.resize(n_ * 2 * b_);
        reward_.resize(n_ * 2 * b_);
        end_reward_.resize(n_ * 2);
        for (auto & s : next_) {
            s = pick(prng);
        }
        for (auto & r : reward_) {
            r = gauss(prng);
        }
        for (auto & r : end_reward_) {
            r = gauss(prng);
        }
    }

    /// @return the start state, 0
    state_type start() const
    {
        return 0;
    }

    /// @return the terminal state, n
    state_type terminal() const
    {
        return n_;
    }

    /// @return the rules as a model, p(s', r | s, a)
    model_type model() const
    {
        model_type m;
        for (state_type s = 0; s <= n_; s++) {
            m.states.push_back(relearn::state<state_type>(s));
        }
        const unsigned int n = n_;
        m.actions = [n](const relearn::state<state_type> & s) {
            std::vector<relearn::action<action_type>> both;
            if (s.trait() != n) {
                both = {relearn::action<action_type>(0u), relearn::action<action_type>(1u)};
            }
            return both;
        };
        auto self = *this;
        m.outcomes = [self](const relearn::state<state_type> & s,
                            const relearn::action<action_type> & a) {
            std::vector<model_type::outcome> outcomes;
            const unsigned int pair = s.trait() * 2 + a.trait();
            for (unsigned int i = 0; i < self.b_; i++) {
                outcomes.push_back({relearn::state<state_type>(self.next_[pair * self.b_ + i]),
                                    self.reward_[pair * self.b_ + i], 0.9 / self.b_});
            }
            outcomes.push_back({relearn::state<state_type>(self.n_), self.end_reward_[pair], 0.1});
            return outcomes;
        };
        return m;
    }

private:
    // n: the number of states, and b: the branches of each pair
    unsigned int n_;
    unsigned int b_;
    // the next states, and the rewards, of every branch of every pair
    std::vector<state_type> next_;
    std::vector<double> reward_;
    // the reward of ending the episode, from every pair
    std::vector<double> end_reward_;
};
} // namespace games
#endif
