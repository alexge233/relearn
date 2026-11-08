#ifndef RELEARN_WINDOW_LESSON_HPP
#define RELEARN_WINDOW_LESSON_HPP
/**
 * A lesson: a demo that explains itself. It plays a game step by step, and shows what the learner did at each step
 * in its own formula, with the real numbers, and in plain words.
 *
 * Nothing here needs raylib, so that all of it is tested with the rest (have a look at tests/lesson_test.cpp);
 * studio.hpp draws it.
 *
 * - `numbers` are the terms of one update, and `formula_terms` and `plain_words` what is written about them.
 * - A *rule* (`q_learning_rule`, `sarsa_rule`, `expected_sarsa_rule`) says which learner it is, and how that learner
 *   values what comes next: the only place the three differ.
 * - `tabular_session<game_type, rule>` plays and learns one step at a time, behind the `session` interface, so that
 *   a lesson can switch learner whilst it runs.
 * - `controls` is what the keys and the top bar change: pause, one step, the speed, what is shown.
 */
#include "../games/game.hpp"
#include "../games/grid.hpp"
#include "../games/watch.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace games
{
namespace lessons
{
/// @return @param v with two decimals, and never "-0.00"
inline std::string two_decimals(double v)
{
    char text[32];
    std::snprintf(text, sizeof text, "%.2f", std::fabs(v) < 0.005 ? 0.0 : v);
    return text;
}

/**
 * @struct numbers
 * @brief the terms of one update: Q(s,a) <- Q(s,a) + alpha [ r + gamma next - Q(s,a) ]
 */
struct numbers
{
    /// Q(s,a) before the update
    double believed = 0;
    /// the learning rate
    double alpha = 0;
    /// the reward received
    double reward = 0;
    /// the discount
    double gamma = 0;
    /// the value of what comes next, as this learner sees it (0 if the episode ended)
    double next = 0;
    /// Q(s,a) after the update, read back from the policy the learner changed
    double learnt = 0;
    /// true if the episode ended with this step
    bool ended = false;
    /// true once a step has been taken
    bool valid = false;

    /// @return what the move turned out to be worth: r + gamma next
    double target() const
    {
        return reward + gamma * next;
    }

    /// @return how wrong the belief was: the target minus what was believed
    double surprise() const
    {
        return target() - believed;
    }
};

/// what a term of the formula stands for, which decides its colour
enum class role
{
    plain,
    result,
    belief,
    speed,
    reward,
    future,
    next
};

/**
 * @struct term
 * @brief one column of the formula: the symbol, its value at this step, and what it means
 */
struct term
{
    /// the symbol: "alpha", "gamma" and "arrow" are drawn as glyphs
    std::string symbol;
    /// its value at this step
    std::string number;
    /// what it means, in a word or two
    std::string label;
    /// what it stands for
    role kind;
};

/// @return the formula as columns, with the numbers of @param n; @param next_symbol and @param next_label are the rule's
inline std::vector<term> formula_terms(const numbers & n,
                                       const std::string & next_symbol,
                                       const std::string & next_label)
{
    return {{"Q(s,a)", two_decimals(n.learnt), "new belief", role::result},
            {"arrow", "=", "", role::plain},
            {"Q(s,a)", two_decimals(n.believed), "I believed", role::belief},
            {"+", "+", "", role::plain},
            {"alpha", two_decimals(n.alpha), "speed", role::speed},
            {"[", "[", "", role::plain},
            {"r", two_decimals(n.reward), "reward", role::reward},
            {"+", "+", "", role::plain},
            {"gamma", two_decimals(n.gamma), "future", role::future},
            {next_symbol, two_decimals(n.next), next_label, role::next},
            {"-", "-", "", role::plain},
            {"Q(s,a)", two_decimals(n.believed), "I believed", role::belief},
            {"]", "]", "", role::plain}};
}

/**
 * @struct column
 * @brief where a term of the formula goes
 */
struct column
{
    float x;
    float width;
};

/**
 * @brief place the terms in columns, centred in @param available
 * @param widths is the width of each term
 * @param left is the left edge of the room, @param available its width
 * @param gap is the room between two columns, reduced if the formula would not fit
 */
inline std::vector<column> formula_columns(const std::vector<float> & widths,
                                           float left,
                                           float available,
                                           float gap)
{
    float total = 0;
    for (auto w : widths) {
        total += w;
    }
    const float count = static_cast<float>(widths.size());
    if (total + gap * count > available) {
        gap = std::max(0.0f, (available - total) / count);
    }
    float x = left + (available - total - gap * count) / 2;
    std::vector<column> out;
    for (auto w : widths) {
        out.push_back({x, w + gap});
        x += w + gap;
    }
    return out;
}

/**
 * @brief the update in plain words
 * @param n are the numbers of the update
 * @param action is the name of the move
 * @param next_words is how the rule says what comes next, as "Best it can do from the next cell"
 * @return four sentences
 */
inline std::vector<std::string> plain_words(const numbers & n,
                                            const std::string & action,
                                            const std::string & next_words)
{
    std::vector<std::string> lines;
    lines.push_back("It believed going " + action + " from here was worth " + two_decimals(n.believed) + ".");
    lines.push_back(n.ended ? "It went. Reward " + two_decimals(n.reward) + ". The episode is over, so nothing comes after."
                            : "It went. Reward " + two_decimals(n.reward) + ". " + next_words + ": " + two_decimals(n.next) + ".");
    const bool none = std::fabs(n.surprise()) < 0.005;
    lines.push_back("So the move was really worth " + two_decimals(n.target()) + ". " +
                    (none ? std::string("No surprise: it was right.") : "Surprise: " + two_decimals(n.surprise()) + "."));
    lines.push_back("It moves its belief " + std::to_string(static_cast<int>(n.alpha * 100 + 0.5)) +
                    "% of the way there: now " + two_decimals(n.learnt) + ".");
    return lines;
}

/// @return the value of what comes next if a policy only stores the actions tried: the best stored, 0 if none is
template <class game_type>
double best_stored(const typename types<game_type>::policy & memory,
                   const typename game_type::state_type & s)
{
    const double best = memory.best_value(typename types<game_type>::state(s));
    return std::isnan(best) ? 0.0 : best;
}

/**
 * @struct q_learning_rule
 * @brief Q-learning: what comes next is valued as the best move from there, whatever the agent then does
 */
template <class game_type>
struct q_learning_rule
{
    using learner_type = relearn::q_learning<typename types<game_type>::state, typename types<game_type>::action>;

    static const char * name() { return "Q-learning"; }
    static const char * next_symbol() { return "max Q(s',a')"; }
    static const char * next_label() { return "best from there"; }
    static const char * next_words() { return "Best it can do from the next cell"; }
    static const char * idea() { return "Q-learning assumes its next move will be the best one, even when it is about to explore."; }

    static void prepare(learner_type &,
                        const typename types<game_type>::policy &,
                        const game_type &,
                        const double &)
    {}

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
 * @struct sarsa_rule
 * @brief SARSA: what comes next is valued as the move the agent has actually chosen
 */
template <class game_type>
struct sarsa_rule
{
    using learner_type = relearn::sarsa<typename types<game_type>::state, typename types<game_type>::action>;

    static const char * name() { return "SARSA"; }
    static const char * next_symbol() { return "Q(s',a')"; }
    static const char * next_label() { return "the next move"; }
    static const char * next_words() { return "The move it chose next is worth"; }
    static const char * idea() { return "SARSA values the move it will really make, slips included, so it learns to keep away from danger."; }

    static void prepare(learner_type &,
                        const typename types<game_type>::policy &,
                        const game_type &,
                        const double &)
    {}

    static double next_value(const typename types<game_type>::policy & memory,
                             const typename game_type::state_type & next,
                             const typename game_type::action_type & chosen,
                             const std::vector<typename game_type::action_type> &,
                             double)
    {
        return memory.value(typename types<game_type>::state(next), typename types<game_type>::action(chosen));
    }
};

/**
 * @struct expected_sarsa_rule
 * @brief Expected SARSA: what comes next is valued as the average over the moves the agent might make
 */
template <class game_type>
struct expected_sarsa_rule
{
    using learner_type = relearn::expected_sarsa<typename types<game_type>::state, typename types<game_type>::action>;

    static const char * name() { return "Expected SARSA"; }
    static const char * next_symbol() { return "E[Q(s',a')]"; }
    static const char * next_label() { return "average next move"; }
    static const char * next_words() { return "On average, the next move is worth"; }
    static const char * idea() { return "Expected SARSA averages over every move it might make next, so one unlucky slip counts for less."; }

    static void prepare(learner_type & learner,
                        const typename types<game_type>::policy & memory,
                        const game_type & game,
                        const double & epsilon)
    {
        learner.probabilities = [&memory, game, &epsilon](const typename types<game_type>::state & s) {
            return epsilon_greedy_probabilities<game_type>(memory, s.trait(), game.actions(s.trait()), epsilon);
        };
    }

    static double next_value(const typename types<game_type>::policy & memory,
                             const typename game_type::state_type & next,
                             const typename game_type::action_type &,
                             const std::vector<typename game_type::action_type> & legal,
                             double epsilon)
    {
        double expected = 0;
        for (const auto & item : epsilon_greedy_probabilities<game_type>(memory, next, legal, epsilon)) {
            expected += item.second * memory.value(typename types<game_type>::state(next), item.first);
        }
        return expected;
    }
};

/**
 * @class session
 * @brief a learner playing a grid game one step at a time, as a lesson sees it
 */
class session
{
public:
    virtual ~session() = default;
    /// @return the learner's name
    virtual std::string name() const = 0;
    /// @return the symbol of what comes next, in the formula
    virtual std::string next_symbol() const = 0;
    /// @return what that symbol means, in a word or two
    virtual std::string next_label() const = 0;
    /// @return how the plain words say what comes next
    virtual std::string next_words() const = 0;
    /// @return what sets this learner apart, in a sentence
    virtual std::string idea() const = 0;
    /// @brief play and learn one step
    virtual void step() = 0;
    /// @brief give up on the episode and start another, keeping what was learnt
    virtual void restart() = 0;
    /// @brief forget everything learnt, and start again
    virtual void reset() = 0;
    /// @return the numbers of the last update
    virtual const numbers & latest() const = 0;
    /// @return the cell the agent is in
    virtual position at() const = 0;
    /// @return the cell the last step started from
    virtual position from() const = 0;
    /// @return the move of the last step
    virtual unsigned int last_action() const = 0;
    /// @return true if the last step was punished: more than a step's cost
    virtual bool punished() const = 0;
    /// @return true if some action of @param p has been learnt
    virtual bool known(const position & p) const = 0;
    /// @return the value of move @param a in cell @param p (0 if never tried)
    virtual double q(const position & p, unsigned int a) const = 0;
    /// @return the highest value of the moves of @param p
    virtual double best(const position & p) const = 0;
    /// @return the move the agent would choose in @param p when it is not exploring: the first of the best
    virtual unsigned int best_action(const position & p) const = 0;
    /// @return the moves of a cell
    virtual std::vector<unsigned int> moves() const = 0;
    /// @return the return of each episode that ended
    virtual const std::vector<double> & returns() const = 0;
    /// @return the episodes started
    virtual std::size_t episode() const = 0;
    /// @return the steps of this episode
    virtual std::size_t steps() const = 0;
    /// @return the return of this episode so far
    virtual double total() const = 0;
    /// @return true if an episode is under way, and not yet on the curve
    virtual bool playing() const = 0;
    /// @return the chance of a random move
    virtual double epsilon() const = 0;
    /// @return the learning rate
    virtual double alpha() const = 0;
    /// @return the discount
    virtual double gamma() const = 0;
    /// @brief set the learning rate, kept between 0 and 1, from the next step on
    virtual void set_alpha(double value) = 0;
    /// @brief set the discount, kept between 0 and 1, from the next step on
    virtual void set_gamma(double value) = 0;
    /// @brief set the chance of a random move, kept between 0 and 1, from the next move on
    virtual void set_epsilon(double value) = 0;
    /// @brief go back to the values the lesson started with
    virtual void restore_settings() = 0;
};

/**
 * @class tabular_session
 * @brief a tabular learner on a grid game, learning after every step, in the order of `games::play_online`
 * @tparam game_type is a grid game: its states are `position`s, its actions the four moves
 * @tparam rule is the learner: `q_learning_rule`, `sarsa_rule` or `expected_sarsa_rule`
 */
template <class game_type,
          class rule>
class tabular_session : public session
{
public:
    using state  = typename types<game_type>::state;
    using action = typename types<game_type>::action;

    /**
     * @param game is the game
     * @param alpha is the learning rate, @param gamma the discount, @param epsilon the chance of a random move
     * @param seed seeds the random moves
     * @param max_steps cuts an episode short
     */
    tabular_session(game_type game,
                    double alpha,
                    double gamma,
                    double epsilon,
                    unsigned int seed,
                    std::size_t max_steps = 1000)
    : game_(std::move(game)),
      alpha_(alpha),
      gamma_(gamma),
      epsilon_(epsilon),
      first_alpha_(alpha),
      first_gamma_(gamma),
      first_epsilon_(epsilon),
      seed_(seed),
      max_steps_(max_steps),
      prng_(seed)
    {
        make_learner();
    }

    tabular_session(const tabular_session &) = delete;
    tabular_session & operator=(const tabular_session &) = delete;

    std::string name() const override { return rule::name(); }
    std::string next_symbol() const override { return rule::next_symbol(); }
    std::string next_label() const override { return rule::next_label(); }
    std::string next_words() const override { return rule::next_words(); }
    std::string idea() const override { return rule::idea(); }

    void step() override
    {
        if (!playing_) {
            begin();
        }
        numbers n;
        n.alpha    = alpha_;
        n.gamma    = gamma_;
        n.believed = memory_.value(state(s_), action(a_));
        from_ = s_;
        last_ = a_;
        auto out = game_.step(a_);
        n.reward = out.reward;
        steps_++;
        total_ += out.reward;
        typename types<game_type>::episode chain;
        chain.push_back({state(out.reward, s_), action(a_)});
        if (out.terminated) {
            n.ended = true;
            learner_(chain, memory_, true);
        }
        else {
            const auto legal = game_.actions(out.next);
            const auto chosen = choose(out.next, legal);
            n.next = rule::next_value(memory_, out.next, chosen, legal, epsilon_);
            chain.push_back({state(0, out.next), action(chosen)});
            learner_(chain, memory_, false);
            a_ = chosen;
        }
        n.learnt = memory_.value(state(from_), action(last_));
        n.valid  = true;
        latest_  = n;
        s_ = out.next;
        punished_ = out.reward < -1;
        if (out.terminated || steps_ >= max_steps_) {
            returns_.push_back(total_);
            playing_ = false;
        }
    }

    void restart() override
    {
        playing_ = false;
        begin();
        latest_ = numbers();
    }

    void reset() override
    {
        memory_ = typename types<game_type>::policy();
        make_learner();
        prng_.seed(seed_);
        returns_.clear();
        episode_ = 0;
        playing_ = false;
        begin();
        latest_ = numbers();
    }

    const numbers & latest() const override { return latest_; }
    position at() const override { return s_; }
    position from() const override { return from_; }
    unsigned int last_action() const override { return last_; }
    bool punished() const override { return punished_; }

    bool known(const position & p) const override
    {
        return !std::isnan(memory_.best_value(state(p)));
    }

    double q(const position & p,
             unsigned int a) const override
    {
        return memory_.value(state(p), action(a));
    }

    double best(const position & p) const override
    {
        return q(p, best_action(p));
    }

    unsigned int best_action(const position & p) const override
    {
        const auto legal = game_.actions(p);
        unsigned int chosen = legal.front();
        for (auto a : legal) {
            if (q(p, a) > q(p, chosen)) {
                chosen = a;
            }
        }
        return chosen;
    }

    std::vector<unsigned int> moves() const override { return game_.actions(s_); }
    const std::vector<double> & returns() const override { return returns_; }
    std::size_t episode() const override { return episode_; }
    std::size_t steps() const override { return steps_; }
    double total() const override { return total_; }
    bool playing() const override { return playing_; }
    double epsilon() const override { return epsilon_; }
    double alpha() const override { return alpha_; }
    double gamma() const override { return gamma_; }

    void set_alpha(double value) override
    {
        alpha_ = unit(value);
        learner_.alpha = alpha_;
    }

    void set_gamma(double value) override
    {
        gamma_ = unit(value);
        learner_.gamma = gamma_;
    }

    void set_epsilon(double value) override
    {
        epsilon_ = unit(value);
    }

    void restore_settings() override
    {
        set_alpha(first_alpha_);
        set_gamma(first_gamma_);
        set_epsilon(first_epsilon_);
    }

    /// @return the policy being learnt
    const typename types<game_type>::policy & memory() const
    {
        return memory_;
    }

private:
    /// @return @param value kept between 0 and 1
    static double unit(double value)
    {
        return std::max(0.0, std::min(1.0, value));
    }

    /// @brief a new learner, set up for this policy
    void make_learner()
    {
        learner_ = typename rule::learner_type();
        learner_.alpha = alpha_;
        learner_.gamma = gamma_;
        rule::prepare(learner_, memory_, game_, epsilon_);
    }

    /// @return the agent's move in @param s: epsilon-greedy
    typename game_type::action_type choose(const typename game_type::state_type & s,
                                           const std::vector<typename game_type::action_type> & legal)
    {
        return epsilon_greedy<game_type>(memory_, s, legal, epsilon_, prng_);
    }

    /// @brief start an episode
    void begin()
    {
        s_ = game_.reset();
        from_ = s_;
        a_ = choose(s_, game_.actions(s_));
        episode_++;
        steps_ = 0;
        total_ = 0;
        punished_ = false;
        playing_ = true;
    }

    game_type game_;
    double alpha_;
    double gamma_;
    double epsilon_;
    double first_alpha_;
    double first_gamma_;
    double first_epsilon_;
    unsigned int seed_;
    std::size_t max_steps_;
    std::mt19937 prng_;
    typename types<game_type>::policy memory_;
    typename rule::learner_type learner_;
    typename game_type::state_type s_{};
    typename game_type::state_type from_{};
    typename game_type::action_type a_{};
    typename game_type::action_type last_{};
    numbers latest_;
    std::vector<double> returns_;
    std::size_t episode_ = 0;
    std::size_t steps_ = 0;
    double total_ = 0;
    bool punished_ = false;
    bool playing_ = false;
};

/// what the keys and the top bar can ask for
enum class command
{
    none,
    learner,
    restart,
    reset,
    pause,
    step,
    slower,
    faster,
    values,
    arrows,
    intro,
    sound,
    help,
    quit
};

/// the key codes that are not their own character
enum : int
{
    key_tab = 258,
    key_right = 262,
    key_keypad_minus = 333,
    key_keypad_plus = 334
};

/// @return the command of the key with code @param key (letters are upper case), `command::none` if it has none
inline command command_of_key(int key)
{
    switch (key) {
        case ' ':  return command::pause;
        case 'N':
        case key_right: return command::step;
        case 'R':  return command::restart;
        case 'Z':  return command::reset;
        case '-':
        case key_keypad_minus: return command::slower;
        case '=':
        case key_keypad_plus: return command::faster;
        case 'V':  return command::values;
        case 'A':  return command::arrows;
        case 'I':  return command::intro;
        case 'M':  return command::sound;
        case 'H':  return command::help;
        case 'Q':  return command::quit;
        case key_tab: return command::learner;
        default:   return command::none;
    }
}

/// the sounds a step can make
enum class cue
{
    none,
    fall,
    treasure
};

/// @return the sound of a step: @param punished is whether it was punished, @param ended whether it ended the episode
inline cue cue_of(bool punished,
                  bool ended)
{
    return punished ? cue::fall : (ended ? cue::treasure : cue::none);
}

/**
 * @class controls
 * @brief how the lesson is being run: the speed, whether it is paused, what is shown
 */
class controls
{
public:
    /// true whilst paused
    bool paused = false;
    /// show the colours of the values
    bool values = true;
    /// show the arrows of the policy
    bool arrows = true;
    /// the help is over the screen
    bool help = false;
    /// the music and the effects are on
    bool sound = true;

    /// @return the speeds, in steps per second
    static const std::vector<double> & speeds()
    {
        static const std::vector<double> table = {1, 2, 5, 10, 30, 120, 600, 3000};
        return table;
    }

    /// @return the speed, in steps per second
    double rate() const
    {
        return speeds()[index_];
    }

    /// @brief go one speed down, stopping at the slowest
    void slower()
    {
        if (index_ > 0) {
            index_--;
        }
    }

    /// @brief go one speed up, stopping at the fastest
    void faster()
    {
        if (index_ + 1 < speeds().size()) {
            index_++;
        }
    }

    /// @return true if the agent is slow enough to be seen moving between two cells
    bool gliding() const
    {
        return rate() <= 10;
    }

    /// @return true if a step is slow enough to have its own sound: above that, it would be a constant rattle
    bool audible_steps() const
    {
        return paused || gliding();
    }

    /// @brief ask for one step, which also pauses
    void single_step()
    {
        paused = true;
        pending_++;
    }

    /**
     * @brief the steps to play in a frame
     * @param seconds is the length of the frame
     * @return how many steps: those asked for one by one if paused, else as many as the speed has built up
     */
    unsigned int steps_for(double seconds)
    {
        if (paused) {
            const unsigned int asked = pending_;
            pending_ = 0;
            return asked;
        }
        pending_ = 0;
        due_ += seconds * rate();
        const auto whole = static_cast<unsigned int>(due_);
        due_ -= whole;
        return whole;
    }

    /// @return how far the agent is between its last two cells, from 0 to 1, whilst gliding
    double progress() const
    {
        return paused || !gliding() ? 1.0 : std::min(1.0, due_);
    }

private:
    std::size_t index_ = 2;
    double due_ = 0;
    unsigned int pending_ = 0;
};

/**
 * @struct menu_item
 * @brief an entry of the top bar, and where it is
 */
struct menu_item
{
    command what;
    std::string label;
    float x;
    float width;
};

/**
 * @brief lay out the top bar
 * @param entries are the commands and their labels
 * @param measure gives the width of a label
 * @param left is where the first starts, @param padding the room each side of a label
 */
template <class measuring>
std::vector<menu_item> lay_out_menu(const std::vector<std::pair<command, std::string>> & entries,
                                    const measuring & measure,
                                    float left,
                                    float padding)
{
    std::vector<menu_item> items;
    float x = left;
    for (const auto & entry : entries) {
        const float width = measure(entry.second) + 2 * padding;
        items.push_back({entry.first, entry.second, x, width});
        x += width;
    }
    return items;
}

/// @return the command at the point (@param x, @param y) of a bar of height @param height, `command::none` if none
inline command menu_hit(const std::vector<menu_item> & items,
                        float x,
                        float y,
                        float height)
{
    if (y < 0 || y >= height) {
        return command::none;
    }
    for (const auto & item : items) {
        if (x >= item.x && x < item.x + item.width) {
            return item.what;
        }
    }
    return command::none;
}

/// @return @param text cut into lines of at most @param columns characters, at the spaces
inline std::vector<std::string> wrap_text(const std::string & text,
                                          std::size_t columns)
{
    std::vector<std::string> lines;
    std::string line;
    std::size_t at = 0;
    while (at < text.size()) {
        std::size_t end = text.find(' ', at);
        if (end == std::string::npos) {
            end = text.size();
        }
        const std::string word = text.substr(at, end - at);
        if (!line.empty() && line.size() + 1 + word.size() > columns) {
            lines.push_back(line);
            line.clear();
        }
        line += (line.empty() ? "" : " ") + word;
        at = end + 1;
    }
    if (!line.empty()) {
        lines.push_back(line);
    }
    return lines;
}

/// @return the colour of @param t, from 0 (worst) to 1 (best): deep blue, violet, orange, gold
inline rgb heat_colour(double t)
{
    static const rgb stops[4] = {{30, 50, 170}, {140, 60, 170}, {240, 120, 60}, {255, 225, 90}};
    t = std::max(0.0, std::min(1.0, t));
    const double place = t * 3;
    const int low = std::min(2, static_cast<int>(place));
    return gradient(place - low, stops[low], stops[low + 1]);
}

/// @return @param v placed between @param low and @param high, from 0 to 1 (1 if they are equal)
inline double scaled(double v,
                     double low,
                     double high)
{
    return high > low ? (v - low) / (high - low) : 1.0;
}

/// @return the rows of the pixel glyph of @param symbol ("alpha", "gamma", "epsilon", "arrow"), empty if it has none
inline std::vector<std::string> glyph(const std::string & symbol)
{
    if (symbol == "alpha") {
        return {"..........",
                "..........",
                "..........",
                "..###...##",
                ".#...#.#..",
                "#.....##..",
                "#.....#...",
                "#.....#...",
                "#.....##..",
                ".#...#.#..",
                "..###...##"};
    }
    if (symbol == "gamma") {
        return {"..........",
                "..........",
                "..........",
                "##......##",
                ".#......#.",
                "..#....#..",
                "..#....#..",
                "...#..#...",
                "...#..#...",
                "....##....",
                "....##....",
                "...#..#...",
                "...#..#...",
                "....##...."};
    }
    if (symbol == "epsilon") {
        return {"........",
                "........",
                "........",
                "..#####.",
                ".#......",
                "#.......",
                "######..",
                "#.......",
                "#.......",
                ".#......",
                "..#####."};
    }
    if (symbol == "arrow") {
        return {"............",
                "............",
                "............",
                "............",
                "....#.......",
                "...##.......",
                "..##........",
                ".###########",
                "..##........",
                "...##.......",
                "....#......."};
    }
    return {};
}

/// @return how far a crawl has scrolled after @param seconds, at @param speed of its height per second, from 0
inline double crawl_position(double seconds,
                             double speed)
{
    return std::max(0.0, seconds) * speed;
}

/// @return the lowest of @param values, but no lower than @param floor: so that a few awful first episodes do not
///         squash the rest of a learning curve
inline double lowest_shown(const std::vector<double> & values,
                           double floor)
{
    double lowest = 0;
    bool first = true;
    for (auto v : values) {
        if (first || v < lowest) {
            lowest = v;
            first = false;
        }
    }
    return std::max(lowest, floor);
}

/**
 * @brief the points of a learning curve
 * @param values are the returns of the episodes, oldest first
 * @param columns is the most points there is room for
 * @return @param values, or, if there are more of them than columns, the mean of those falling in each column
 */
inline std::vector<double> curve_points(const std::vector<double> & values,
                                        std::size_t columns)
{
    if (values.size() <= columns || columns == 0) {
        return values;
    }
    std::vector<double> sums(columns, 0);
    std::vector<std::size_t> counts(columns, 0);
    for (std::size_t i = 0; i < values.size(); i++) {
        const std::size_t column = i * columns / values.size();
        sums[column] += values[i];
        counts[column]++;
    }
    for (std::size_t c = 0; c < columns; c++) {
        sums[c] /= static_cast<double>(counts[c]);
    }
    return sums;
}

/// @return the points of the learning curve now: the returns so far, and, if an episode is under way, what it has
///         earned until now, so that the curve moves from the very first step
inline std::vector<double> live_curve(const session & now)
{
    std::vector<double> values = now.returns();
    if (now.playing() && now.steps() > 0) {
        values.push_back(now.total());
    }
    return values;
}

/**
 * @struct slider_range
 * @brief the values a slider goes through
 */
struct slider_range
{
    double low;
    double high;
    double step;
};

/**
 * @brief the value of a slider at a point of its track
 * @param range are its values
 * @param mouse_x is where the mouse is
 * @param left and @param width are the track's
 * @return the value there, on one of its steps, inside its range even with the mouse beyond the track's ends
 */
inline double slider_value(const slider_range & range,
                           float mouse_x,
                           float left,
                           float width)
{
    const double along = std::max(0.0, std::min(1.0, static_cast<double>(mouse_x - left) / width));
    const double raw = range.low + along * (range.high - range.low);
    const double snapped = range.low + std::round((raw - range.low) / range.step) * range.step;
    return std::max(range.low, std::min(range.high, snapped));
}

/// @return how far along its track, from 0 to 1, a slider of @param range with @param value sits
inline double slider_fraction(const slider_range & range,
                              double value)
{
    return std::max(0.0, std::min(1.0, (value - range.low) / (range.high - range.low)));
}

/**
 * @struct hot_zone
 * @brief a part of the screen that explains itself when the mouse rests on it
 */
struct hot_zone
{
    float x;
    float y;
    float width;
    float height;
    /// what the bubble says
    std::string text;
};

/// @return the zone under (@param x, @param y), the last one given if several overlap, or null if none
inline const hot_zone * zone_at(const std::vector<hot_zone> & zones,
                                float x,
                                float y)
{
    const hot_zone * found = nullptr;
    for (const auto & zone : zones) {
        if (x >= zone.x && x < zone.x + zone.width && y >= zone.y && y < zone.y + zone.height) {
            found = &zone;
        }
    }
    return found;
}

/**
 * @struct bubble_place
 * @brief where a help bubble goes
 */
struct bubble_place
{
    float x;
    float y;
};

/**
 * @brief place a bubble of @param width x @param height near the mouse at (@param mouse_x, @param mouse_y)
 * @param screen_width and @param screen_height are the size of the window
 * @return below and to the right of the mouse; on its other side if it would leave the window that way, and always
 *         inside the window
 */
inline bubble_place place_bubble(float mouse_x,
                                 float mouse_y,
                                 float width,
                                 float height,
                                 float screen_width,
                                 float screen_height)
{
    const float offset = 16;
    const float margin = 6;
    float x = mouse_x + offset;
    float y = mouse_y + offset;
    if (x + width > screen_width - margin) {
        x = mouse_x - offset - width;
    }
    if (y + height > screen_height - margin) {
        y = mouse_y - offset - height;
    }
    return {std::max(margin, std::min(x, screen_width - margin - width)),
            std::max(margin, std::min(y, screen_height - margin - height))};
}

/**
 * @class hover_timer
 * @brief a bubble appears once the mouse has rested on the same thing for a moment, so that it does not flicker
 */
class hover_timer
{
public:
    /// the seconds the mouse must rest
    static constexpr double wait = 0.3;

    /**
     * @brief a frame has gone by
     * @param zone identifies the zone under the mouse, empty if there is none
     * @param seconds is the length of the frame
     * @return true if its bubble is to be shown
     */
    bool update(const std::string & zone,
                double seconds)
    {
        if (zone != zone_) {
            zone_ = zone;
            rested_ = 0;
        }
        else {
            rested_ += seconds;
        }
        return !zone_.empty() && rested_ >= wait;
    }

private:
    std::string zone_;
    double rested_ = 0;
};

/// @return what a bubble says about the entry @param what of the top bar
inline std::string menu_help(command what)
{
    switch (what) {
        case command::learner: return "Choose how it learns. Each learner has its own memory. (Tab)";
        case command::restart: return "Send the walker back to the start. It keeps everything it has learnt. (R)";
        case command::reset:   return "Forget everything and begin again from nothing. (Z)";
        case command::pause:   return "Stop or carry on, to read a step at your own pace. (Space)";
        case command::step:    return "Take exactly one step, then wait. (N or the right arrow)";
        case command::slower:  return "Fewer steps each second. (-)";
        case command::faster:  return "More steps each second. Very fast, it learns in seconds. (=)";
        case command::values:  return "Show or hide the colours: how good it believes each cell is. (V)";
        case command::arrows:  return "Show or hide the arrows: its best known move in each cell. (A)";
        case command::intro:   return "Watch the introduction again. (I)";
        case command::sound:   return "The music and the sound effects, on or off. (M)";
        case command::help:    return "All the keys, and what is on the screen. (H)";
        default:               return "";
    }
}

/**
 * @brief what a bubble says about a term of the formula
 * @param t is the term
 * @param next_idea is what sets this learner apart, for the term of what comes next
 * @return the explanation, empty for the plain signs
 */
inline std::string term_help(const term & t,
                             const std::string & next_idea)
{
    switch (t.kind) {
        case role::result: return "The new belief: what this move will be worth from now on, the next time it is here.";
        case role::belief: return "The old belief: what it thought this move was worth before taking it.";
        case role::speed:  return "alpha, the learning rate: how far the belief moves. At 0 it never learns; at 1 it forgets the old belief entirely.";
        case role::reward: return "The reward: what this step just paid.";
        case role::future: return "gamma, the discount: how much later rewards count. At 0 only the next reward matters; at 1 all of them count the same.";
        case role::next:   return "What comes next, by its own reckoning. " + next_idea;
        default: break;
    }
    if (t.symbol == "[" || t.symbol == "]") {
        return "Everything inside the brackets is the surprise: what the move turned out to be worth, minus what it believed.";
    }
    if (t.symbol == "arrow") {
        return "Replace the value on the left with what is on the right.";
    }
    return "";
}

/**
 * @brief what a bubble says about a cell of a grid
 * @param now is the session
 * @param p is the cell
 * @param names are the names of the moves
 */
inline std::string cell_help(const session & now,
                             const position & p,
                             const std::vector<std::string> & names)
{
    if (!now.known(p)) {
        return "It has not been here yet, so it knows nothing about this cell.";
    }
    return "Its best known move here is " + names[now.best_action(p)] + ", which it believes is worth " + two_decimals(now.best(p)) + ".";
}

/**
 * @struct info
 * @brief what a lesson says about itself
 */
struct info
{
    /// the name of the demo
    std::string title;
    /// the book it comes from, or where
    std::string source;
    /// the paragraphs of the intro
    std::vector<std::string> intro;
    /// the sentence on what to watch
    std::string watch_for;
};

/**
 * @struct lesson_options
 * @brief the command line of a lesson
 */
struct lesson_options
{
    /// skip the intro
    bool no_intro = false;
    /// start with the sound off
    bool mute = false;
    /// a directory to render the pictures to, without showing a window, and end (empty: run normally)
    std::string capture;

    /// @brief read the options from the command line
    static lesson_options parse(int argc,
                                char ** argv)
    {
        lesson_options options;
        for (int i = 1; i < argc; i++) {
            const std::string arg = argv[i];
            if (arg == "--no-intro") {
                options.no_intro = true;
            }
            else if (arg == "--mute") {
                options.mute = true;
            }
            else if (arg == "--capture" && i + 1 < argc) {
                options.capture = argv[++i];
            }
        }
        return options;
    }
};
} // namespace lessons
} // namespace games
#endif
