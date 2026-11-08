#ifndef RELEARN_WINDOW_PLANNING_HPP
#define RELEARN_WINDOW_PLANNING_HPP
/**
 * The lessons' planning learner: Dyna-Q, one step at a time, so that each imagined step can be seen. The learner is the
 * library's `relearn::dyna_q`: what is shown is told by its `on_imagined`, and nothing here computes an update. No
 * graphics here either.
 */
#include "lesson.hpp"
#include <string>
#include <vector>

namespace games
{
namespace lessons
{
/**
 * @struct dyna_rule
 * @brief Dyna-Q's real steps: Q-learning, with the planning left for the session to run one step at a time
 */
template <class game_type>
struct dyna_rule
{
    using learner_type = relearn::dyna_q<typename types<game_type>::state, typename types<game_type>::action>;

    static const char * name() { return "Dyna-Q"; }
    static const char * next_symbol() { return "max Q(s',a')"; }
    static const char * next_label() { return "best from there"; }
    static const char * next_words() { return "Best it can do from the next cell"; }
    static const char * idea() { return "Dyna-Q learns from what really happened, then replays what it remembers, with the very same formula."; }

    static void prepare(learner_type & learner,
                        const typename types<game_type>::policy &,
                        const game_type &,
                        const double &)
    {
        learner.planning_steps = 0;
    }

    static double next_value(const typename types<game_type>::policy & memory,
                             const typename game_type::state_type & next,
                             const typename game_type::action_type &,
                             const std::vector<typename game_type::action_type> &,
                             double)
    {
        return best_stored<game_type>(memory, next);
    }
};

/**
 * @class dyna_session
 * @brief Dyna-Q on a grid game: each call of `step` is either a real step or one of the imagined steps after it
 */
template <class game_type>
class dyna_session : public tabular_session<game_type, dyna_rule<game_type>>,
                     public planner
{
public:
    using base = tabular_session<game_type, dyna_rule<game_type>>;

    /**
     * @param game is the game
     * @param alpha is the learning rate, @param gamma the discount, @param epsilon the chance of a random move
     * @param planning is n, the steps imagined after each real one
     * @param seed seeds the random moves
     * @param max_steps cuts an episode short
     */
    dyna_session(game_type game,
                 double alpha,
                 double gamma,
                 double epsilon,
                 unsigned int planning,
                 unsigned int seed,
                 std::size_t max_steps = 3000)
    : base(std::move(game), alpha, gamma, epsilon, seed, max_steps),
      planning_(planning),
      first_planning_(planning)
    {
        listen();
    }

    std::string name() const override
    {
        return (planning_ == 0 ? std::string("Q-learning") : std::string("Dyna-Q")) + ", n = " + std::to_string(planning_);
    }

    std::string idea() const override
    {
        return planning_ == 0 ? "With n = 0 it imagines nothing: this is plain Q-learning. Compare how many episodes it needs."
                              : base::idea();
    }

    void step() override
    {
        if (pending_ > 0) {
            imagine();
            return;
        }
        imagining_ = false;
        imagined_count_ = 0;
        base::step();
        real_steps_++;
        if (!this->playing_) {
            moves_.push_back(static_cast<double>(this->steps_));
        }
        pending_ = planning_;
    }

    void restart() override
    {
        base::restart();
        forget_the_step();
    }

    void reset() override
    {
        base::reset();
        listen();
        forget_the_step();
        moves_.clear();
        real_steps_ = 0;
        imagined_steps_ = 0;
    }

    void restore_settings() override
    {
        base::restore_settings();
        planning_ = first_planning_;
    }

    const std::vector<double> & curve() const override { return moves_; }
    double curve_now() const override { return static_cast<double>(this->steps_); }
    std::string curve_label() const override { return "moves to the treasure, per episode"; }
    std::string curve_help() const override { return "How many real moves each episode took. It should fall as it learns: the more it imagines, the sooner."; }
    std::string tally() const override { return "imagined " + std::to_string(imagined_steps_); }
    planner * plans() override { return this; }
    const planner * plans() const override { return this; }

    bool imagining() const override { return imagining_; }
    position imagined_to() const override { return imagined_to_; }
    unsigned int imagined_count() const override { return imagined_count_; }
    unsigned int planning_steps() const override { return planning_; }
    void set_planning_steps(unsigned int n) override { planning_ = n; }
    std::size_t real_steps() const override { return real_steps_; }
    std::size_t imagined_steps() const override { return imagined_steps_; }

    bool remembers(const position & p) const override
    {
        for (auto a : this->game_.actions(p)) {
            if (this->learner_.remembers(typename base::state(p), typename base::action(a))) {
                return true;
            }
        }
        return false;
    }

private:
    /// @brief show what each imagined step of the learner does (a new learner has to be told again)
    void listen()
    {
        this->learner_.on_imagined = [this](const typename dyna_rule<game_type>::learner_type::imagined & step) {
            numbers n;
            n.alpha    = this->alpha_;
            n.gamma    = this->gamma_;
            n.believed = step.before;
            n.reward   = step.reward;
            n.next     = step.next_value;
            n.ended    = step.terminal;
            n.learnt   = step.after;
            n.valid    = true;
            this->latest_ = n;
            this->from_   = step.state.trait();
            this->last_   = step.action.trait();
            imagined_to_  = step.terminal ? moved_to(this->from_, this->last_) : step.next.trait();
        };
    }

    /// @brief imagine one step: the library's own planning, one at a time
    void imagine()
    {
        this->learner_.planning_steps = 1;
        this->learner_.plan(this->memory_);
        this->learner_.planning_steps = 0;
        imagining_ = true;
        imagined_count_++;
        imagined_steps_++;
        pending_--;
    }

    /// @brief nothing is left to imagine
    void forget_the_step()
    {
        pending_ = 0;
        imagining_ = false;
        imagined_count_ = 0;
    }

    unsigned int planning_;
    unsigned int first_planning_;
    unsigned int pending_ = 0;
    unsigned int imagined_count_ = 0;
    bool imagining_ = false;
    position imagined_to_{};
    std::vector<double> moves_;
    std::size_t real_steps_ = 0;
    std::size_t imagined_steps_ = 0;
};

/// @return the title of the formula's panel, which says whether this step was imagined
inline std::string formula_title(bool imagining,
                                 unsigned int count,
                                 unsigned int of)
{
    return imagining ? "The same formula, from memory: imagined step " + std::to_string(count) + " of " + std::to_string(of)
                     : "The whole idea, with the numbers from this step";
}

/**
 * @brief the plain words of an imagined step
 * @param n is its numbers
 * @param action is the name of its move
 * @param next_words is how the learner says what comes next
 * @return its sentences: those of a real step, but for where the facts came from
 */
inline std::vector<std::string> imagined_words(const numbers & n,
                                               const std::string & action,
                                               const std::string & next_words)
{
    auto lines = plain_words(n, action, next_words);
    lines[0] = "It did not move. It picked something it remembers: going " + action + " from there, believed worth " + two_decimals(n.believed) + ".";
    lines[1] = n.ended ? "It remembers: reward " + two_decimals(n.reward) + ", and the episode ended there."
                       : "It remembers: reward " + two_decimals(n.reward) + ". " + next_words + ": " + two_decimals(n.next) + ".";
    return lines;
}
} // namespace lessons
} // namespace games
#endif
