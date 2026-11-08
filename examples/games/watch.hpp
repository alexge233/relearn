#ifndef RELEARN_GAMES_WATCH_HPP
#define RELEARN_GAMES_WATCH_HPP
/**
 * Watch the agents play, in the terminal.
 *
 * A game is drawn as small pixels: each character cell of the terminal holds two of them, one above the other,
 * with the upper half block `▀` coloured by the top pixel and its background by the bottom one. A character cell is
 * about twice as tall as it is wide, so the pixels come out roughly square.
 *
 *     games::watched<games::cliff_walking> game{games::cliff_walking()};
 *     game.memory = &memory;      // the learnt values, as the colours of the cells (optional)
 *     game.delay  = 30;           // milliseconds between two frames
 *     games::play_online(game, agent, learner, memory, 1000);
 *
 * `watched` is a game like any other (`reset()`, `step()` and `actions()`), which draws the game it wraps after every
 * step, so `play` and `play_online` need no change. With `on` false it draws nothing, and the game plays exactly as
 * it would without it.
 *
 * Each game that can be watched specialises `view<game_type>`, which says how big the picture is and how to draw a
 * state of the game (have a look at views.hpp). The colours are written for a terminal with 24-bit colour, or, when
 * the `COLORTERM` variable does not say `truecolor` or `24bit`, with the nearest of the 256-colour palette.
 *
 * The screen is cleared when the first frame is drawn, and each frame is drawn over the one before from the top of
 * the screen, so the picture (with its status line) must fit the terminal: given the terminal's height
 * (`watch_options::apply` reads it), the learning curve is dropped if it does not fit with it, and a note is written
 * if it does not fit at all.
 */
#include "game.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace games
{
/**
 * @struct rgb
 * @brief a colour, as red, green and blue from 0 to 255
 */
struct rgb
{
    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;

    /// @brief two colours are equal when their three components are
    bool operator==(const rgb & rhs) const
    {
        return (this->r == rhs.r) &&
               (this->g == rhs.g) &&
               (this->b == rhs.b);
    }

    /// @brief two colours differ when a component does
    bool operator!=(const rgb & rhs) const
    {
        return !(*this == rhs);
    }
};

/// the colours the games are drawn with
namespace colours
{
constexpr rgb black  = {0, 0, 0};
constexpr rgb white  = {255, 255, 255};
constexpr rgb grey   = {90, 90, 90};
constexpr rgb floor  = {40, 40, 48};
constexpr rgb wall   = {120, 120, 130};
constexpr rgb agent  = {255, 220, 40};
constexpr rgb start  = {60, 160, 80};
constexpr rgb goal   = {60, 200, 230};
constexpr rgb danger = {200, 50, 40};
constexpr rgb ice    = {170, 200, 235};
constexpr rgb cold   = {30, 40, 120};
constexpr rgb hot    = {240, 180, 40};
constexpr rgb curve  = {120, 230, 120};
constexpr rgb axis   = {70, 70, 80};
constexpr rgb sky    = {20, 24, 40};
constexpr rgb ground = {80, 140, 70};
} // namespace colours

/**
 * @brief the colour @param t of the way from @param from to @param to
 * @param t is clamped to 0 ... 1
 */
inline rgb gradient(double t,
                    const rgb & from,
                    const rgb & to)
{
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    auto mix = [t](unsigned char a, unsigned char b) {
        return static_cast<unsigned char>(a + (static_cast<double>(b) - a) * t + 0.5);
    };
    return {mix(from.r, to.r), mix(from.g, to.g), mix(from.b, to.b)};
}

/**
 * @struct truecolour
 * @brief how a colour is written for a terminal with 24-bit colour: ESC[38;2;r;g;bm and ESC[48;2;r;g;bm
 */
struct truecolour
{
    /// @return true if the terminal says it has 24-bit colour (`COLORTERM` is `truecolor` or `24bit`)
    static bool available()
    {
        return available(std::getenv("COLORTERM"));
    }

    /// @return true if @param colorterm, the value of `COLORTERM` (or null, if it is not set), says `truecolor` or `24bit`
    static bool available(const char * colorterm)
    {
        return colorterm && (std::strcmp(colorterm, "truecolor") == 0 || std::strcmp(colorterm, "24bit") == 0);
    }

    /// @return the escape sequence setting @param c as the colour of the text
    static std::string foreground(const rgb & c)
    {
        return "\x1b[38;2;" + std::to_string(c.r) + ";" + std::to_string(c.g) + ";" + std::to_string(c.b) + "m";
    }

    /// @return the escape sequence setting @param c as the colour behind the text
    static std::string background(const rgb & c)
    {
        return "\x1b[48;2;" + std::to_string(c.r) + ";" + std::to_string(c.g) + ";" + std::to_string(c.b) + "m";
    }
};

/**
 * @struct colour_256
 * @brief how a colour is written for a terminal with the 256-colour palette: the nearest of its 6 x 6 x 6 colours
 */
struct colour_256
{
    /// @return the level, 0 to 5, of the cube's value nearest to @param v: the cube's values are 0, 95, 135, 175,
    ///         215 and 255 (xterm's), so the levels are not evenly spaced
    static unsigned int level(unsigned char v)
    {
        static const unsigned int values[6] = {0, 95, 135, 175, 215, 255};
        unsigned int nearest = 0;
        for (unsigned int i = 1; i < 6; i++) {
            const unsigned int here  = v > values[i] ? v - values[i] : values[i] - v;
            const unsigned int sofar = v > values[nearest] ? v - values[nearest] : values[nearest] - v;
            if (here < sofar) {
                nearest = i;
            }
        }
        return nearest;
    }

    /// @return the index of the palette colour nearest to @param c: 16 + 36 r + 6 g + b, each a level from 0 to 5
    static unsigned int index(const rgb & c)
    {
        return 16 + 36 * level(c.r) + 6 * level(c.g) + level(c.b);
    }

    /// @return the escape sequence setting @param c as the colour of the text
    static std::string foreground(const rgb & c)
    {
        return "\x1b[38;5;" + std::to_string(index(c)) + "m";
    }

    /// @return the escape sequence setting @param c as the colour behind the text
    static std::string background(const rgb & c)
    {
        return "\x1b[48;5;" + std::to_string(index(c)) + "m";
    }
};

/**
 * @class canvas
 * @brief a picture of width x height pixels, drawn on and then rendered for the terminal
 */
class canvas
{
public:
    /// @brief a canvas of @param width x @param height pixels, all of colour @param background
    canvas(unsigned int width,
           unsigned int height,
           const rgb & background = colours::black)
    : width_(width),
      height_(height),
      pixels_(static_cast<std::size_t>(width) * height, background)
    {}

    /// @return the width, in pixels
    unsigned int width() const
    {
        return width_;
    }

    /// @return the height, in pixels
    unsigned int height() const
    {
        return height_;
    }

    /// @return the number of rows of characters it renders to: two pixels per row
    unsigned int rows() const
    {
        return (height_ + 1) / 2;
    }

    /// @brief paint every pixel with @param c
    void clear(const rgb & c)
    {
        std::fill(pixels_.begin(), pixels_.end(), c);
    }

    /// @brief paint the pixel at (@param x, @param y) with @param c; a pixel outside the canvas is ignored
    void plot(int x,
              int y,
              const rgb & c)
    {
        if (inside(x, y)) {
            pixels_[index(x, y)] = c;
        }
    }

    /// @return the colour of the pixel at (@param x, @param y), black outside the canvas
    rgb at(int x,
           int y) const
    {
        return inside(x, y) ? pixels_[index(x, y)] : colours::black;
    }

    /// @brief paint the rectangle of @param w x @param h pixels whose top-left corner is (@param x, @param y)
    void fill(int x,
              int y,
              unsigned int w,
              unsigned int h,
              const rgb & c)
    {
        for (int j = y; j < y + static_cast<int>(h); j++) {
            for (int i = x; i < x + static_cast<int>(w); i++) {
                plot(i, j, c);
            }
        }
    }

    /// @brief draw the straight line from (@param x0, @param y0) to (@param x1, @param y1), both ends included;
    ///        the same pixels whichever end is given first
    void line(int x0,
              int y0,
              int x1,
              int y1,
              const rgb & c)
    {
        // always draw from the left (from the top, if upright), so that the line does not depend on its direction
        if (x1 < x0 || (x1 == x0 && y1 < y0)) {
            std::swap(x0, x1);
            std::swap(y0, y1);
        }
        // Bresenham's algorithm: step along the longer axis, and let the error decide when to step along the other
        const int dx = std::abs(x1 - x0);
        const int dy = -std::abs(y1 - y0);
        const int sx = x0 < x1 ? 1 : -1;
        const int sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        for (;;) {
            plot(x0, y0, c);
            if (x0 == x1 && y0 == y1) {
                break;
            }
            const int twice = 2 * error;
            if (twice >= dy) {
                error += dy;
                x0 += sx;
            }
            if (twice <= dx) {
                error += dx;
                y0 += sy;
            }
        }
    }

    /**
     * @brief render the picture for the terminal, two pixels per character
     * @tparam palette writes the colours: `truecolour` or `colour_256`
     * @return one line per pair of rows of pixels, each ending with the colours reset and a newline;
     *         an odd last row of pixels is drawn over black
     */
    template <class palette>
    std::string render() const
    {
        std::string out;
        for (unsigned int row = 0; row < rows(); row++) {
            const int top    = static_cast<int>(2 * row);
            const int bottom = top + 1;
            for (int x = 0; x < static_cast<int>(width_); x++) {
                out += palette::foreground(at(x, top));
                out += palette::background(at(x, bottom));
                out += "\xE2\x96\x80";
            }
            out += "\x1b[0m\n";
        }
        return out;
    }

private:
    /// @return true if (@param x, @param y) is a pixel of the canvas
    bool inside(int x,
                int y) const
    {
        return x >= 0 && y >= 0 && x < static_cast<int>(width_) && y < static_cast<int>(height_);
    }

    /// @return the index of pixel (@param x, @param y), which must be inside
    std::size_t index(int x,
                      int y) const
    {
        return static_cast<std::size_t>(y) * width_ + static_cast<std::size_t>(x);
    }

    // the size, in pixels
    unsigned int width_;
    unsigned int height_;
    // the pixels, row by row
    std::vector<rgb> pixels_;
};

/**
 * @brief draw a learning curve: one value per episode, as a line from left to right
 * @param c is the canvas, @param x and @param y the top-left corner of the plot, @param w and @param h its size
 * @param values are the values plotted, oldest first; more values than pixels are averaged into the pixels
 * @param colour is the colour of the line (the plot's frame is `colours::axis`)
 *
 * The values are scaled to fit the height, from their lowest to their highest; a single value, or equal ones, are
 * drawn along the middle.
 */
inline void plot_curve(canvas & c,
                       int x,
                       int y,
                       unsigned int w,
                       unsigned int h,
                       const std::vector<double> & values,
                       const rgb & colour = colours::curve)
{
    c.fill(x, y, w, h, colours::sky);
    c.line(x, y + static_cast<int>(h) - 1, x + static_cast<int>(w) - 1, y + static_cast<int>(h) - 1, colours::axis);
    c.line(x, y, x, y + static_cast<int>(h) - 1, colours::axis);
    if (values.empty() || w < 2 || h < 2) {
        return;
    }
    // one point per column: the mean of the values that fall in it
    const unsigned int columns = static_cast<unsigned int>(std::min<std::size_t>(values.size(), w - 1));
    std::vector<double> points(columns, 0);
    std::vector<unsigned int> counts(columns, 0);
    for (std::size_t i = 0; i < values.size(); i++) {
        const std::size_t column = i * columns / values.size();
        points[column] += values[i];
        counts[column]++;
    }
    double lowest  = std::numeric_limits<double>::infinity();
    double highest = -lowest;
    for (unsigned int i = 0; i < columns; i++) {
        points[i] /= counts[i];
        lowest  = std::min(lowest, points[i]);
        highest = std::max(highest, points[i]);
    }
    // the row of each point: the highest at the top, the lowest just above the axis
    auto row = [&](double v) {
        const double span = highest - lowest;
        const double t = span > 0 ? (v - lowest) / span : 0.5;
        return y + static_cast<int>((1 - t) * (h - 2) + 0.5);
    };
    int last_x = x + 1;
    int last_y = row(points[0]);
    for (unsigned int i = 0; i < columns; i++) {
        const int px = x + 1 + static_cast<int>(i);
        const int py = row(points[i]);
        c.line(last_x, last_y, px, py, colour);
        last_x = px;
        last_y = py;
    }
}

/**
 * @struct view
 * @brief how a game is drawn: each game that can be watched specialises it, with
 *
 *     static unsigned int width();     // the picture's size, in pixels
 *     static unsigned int height();
 *     static void draw(canvas & c, const game_type & game, const state_type & at, const policy_type * memory);
 *
 * `at` is the state the game is in, and `memory` the values learnt so far, or null: a view may colour each state by
 * its best value, from `colours::cold` to `colours::hot`.
 */
template <class game_type>
struct view;

/// C++14 has no `std::void_t`: the same, through a struct so that unused parameters still count
template <class...>
struct make_void
{
    using type = void;
};
template <class... types>
using void_t = typename make_void<types...>::type;

/**
 * @struct has_view
 * @brief true if `view<game_type>` is specialised, so that the game can be watched
 */
template <class game_type,
          class = void>
struct has_view : std::false_type
{};
template <class game_type>
struct has_view<game_type, void_t<decltype(view<game_type>::width())>> : std::true_type
{};

/**
 * @struct terminal
 * @brief the escape sequences that place the picture, and the cursor they hide and show again
 */
struct terminal
{
    /// clear the screen
    static const char * clear()
    {
        return "\x1b[2J";
    }

    /// go to the top-left corner, without clearing
    static const char * home()
    {
        return "\x1b[H";
    }

    /// clear from the cursor to the end of the screen
    static const char * clear_below()
    {
        return "\x1b[J";
    }

    /// hide the cursor, so that it does not flicker over the picture
    static const char * hide_cursor()
    {
        return "\x1b[?25l";
    }

    /// show the cursor again
    static const char * show_cursor()
    {
        return "\x1b[?25h";
    }

    /// @return the rows of the terminal the program writes to, or 0 if it is not a terminal (or cannot be asked)
    static unsigned int rows()
    {
#ifndef _WIN32
        struct winsize size;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0) {
            return size.ws_row;
        }
#endif
        return 0;
    }

    /// @brief reset the colours and show the cursor when the program is interrupted (Ctrl-C), and end as the
    ///        interruption would have; only `write` may be called in a signal handler, so it is written with that
    static void restore_on_interrupt()
    {
#ifndef _WIN32
        std::signal(SIGINT, [](int) {
            const char * restore = "\x1b[0m\x1b[?25h\n";
            const ssize_t written = write(STDOUT_FILENO, restore, std::strlen(restore));
            (void) written;
            std::signal(SIGINT, SIG_DFL);
            std::raise(SIGINT);
        });
#endif
    }
};

/**
 * @class terminal_screen
 * @brief where the frames go in the terminal: each one drawn from the top of the screen over the one before, the
 *        cursor hidden meanwhile and shown again at the end, or on Ctrl-C
 *
 * A *screen* is where `watched` shows its frames. Any class with `fits(lines)`, `too_tall(lines)` and
 * `show(canvas, status, title)` is one; the window of the optional raylib sub-project is another.
 */
class terminal_screen
{
public:
    /// the rows of the terminal, if known (0: not checked): `watch_options::apply` reads them
    unsigned int rows = 0;

    /// @brief a screen writing to @param out (the terminal)
    explicit terminal_screen(std::ostream & out = std::cout)
    : out_(&out)
    {}

    /// @brief show the cursor again, below the picture, if any frame was shown
    ~terminal_screen()
    {
        if (shown_) {
            *out_ << terminal::show_cursor() << std::endl;
        }
    }

    /// @return the rows of the terminal a canvas of @param height pixels takes, with the status line under it
    static unsigned int lines_of(unsigned int height)
    {
        return (height + 1) / 2 + 1;
    }

    /// @return the canvas rendered for this terminal: in 24-bit colour if it has it, else in the 256-colour palette
    static std::string render(const canvas & c)
    {
        return truecolour::available() ? c.render<truecolour>() : c.render<colour_256>();
    }

    /// @return true if a frame of @param lines rows fits the terminal, or its height is not known
    bool fits(unsigned int lines) const
    {
        return rows == 0 || lines <= rows;
    }

    /// @brief say, before any frame, that a frame of @param lines rows does not fit the terminal
    void too_tall(unsigned int lines)
    {
        *out_ << "the picture needs " << lines << " rows, and the terminal has " << rows
              << ": make it taller to watch it whole" << std::endl;
    }

    /**
     * @brief show one frame: the canvas and the status line under it, from the top of the screen
     * @return true: a terminal is never closed
     *
     * The first frame hides the cursor and clears the screen. The status line ends with a newline, so that whatever
     * the program prints next goes below it, and the rest of the screen is cleared after it.
     */
    bool show(const canvas & c,
              const std::string & status,
              const std::string &)
    {
        if (!shown_) {
            terminal::restore_on_interrupt();
            *out_ << terminal::hide_cursor() << terminal::clear();
            shown_ = true;
        }
        *out_ << terminal::home() << render(c) << status << "\n" << terminal::clear_below() << std::flush;
        return true;
    }

    /// @return true once a frame has been shown
    bool shown() const
    {
        return shown_;
    }

private:
    // where the frames go
    std::ostream * out_;
    // true once a frame has been shown
    bool shown_ = false;
};

/**
 * @class watched
 * @brief a game that draws itself as it is played: wraps a game, and draws its state after every step
 * @tparam game_type is the game, which must have a `view<game_type>`
 * @tparam screen_type is where the frames go: the terminal, or a window
 *
 * It plays exactly as the game it wraps: `reset()`, `step()` and `actions()` are forwarded, and nothing is drawn
 * (nor written) whilst `on` is false, so an example can watch some episodes and not others.
 */
template <class game_type,
          class screen_type = terminal_screen>
class watched
{
    static_assert(has_view<game_type>::value, "this game has no view<game_type>: it cannot be watched");
public:
    /// the states and actions are those of the wrapped game
    using state_type  = typename game_type::state_type;
    using action_type = typename game_type::action_type;
    /// the values that can be shown as colours
    using policy_type = typename types<game_type>::policy;
    /// the view that draws the game
    using view_type   = view<game_type>;

    /// draw the frames, or play without drawing
    bool on = true;
    /// milliseconds between two frames
    unsigned int delay = 30;
    /// draw one frame every so many steps (1: every step)
    unsigned int every = 1;
    /// the learnt values, shown as the colours of the states, or null
    const policy_type * memory = nullptr;
    /// the name shown in the status line
    std::string title;
    /// the height of the learning curve under the game, in pixels (0: no curve): the curve is dropped if the
    /// frame does not fit the screen with it, and a note is written, once, if the frame does not fit at all
    unsigned int curve_height = 16;

    /**
     * @brief wrap @param game
     * @param screen is where the frames go
     */
    explicit watched(game_type game,
                     screen_type screen = screen_type())
    : game_(std::move(game)),
      screen_(std::move(screen))
    {}

    /**
     * @brief wrap @param game, with the frames written to @param out: a terminal
     */
    watched(game_type game,
            std::ostream & out)
    : game_(std::move(game)),
      screen_(out)
    {}

    /// @return the screen the frames go to
    screen_type & screen()
    {
        return screen_;
    }

    /// @return the wrapped game
    game_type & inner()
    {
        return game_;
    }

    /// @return the wrapped game
    const game_type & inner() const
    {
        return game_;
    }

    /// @return the episodes started so far
    std::size_t episode() const
    {
        return episode_;
    }

    /// @return the steps of the current episode
    std::size_t steps() const
    {
        return steps_;
    }

    /// @return the undiscounted return of the current episode so far
    double total() const
    {
        return total_;
    }

    /// @return the return of each episode that ended (or was left) so far, for the learning curve: an episode's
    ///         return is recorded when it ends, or, if it was cut short, when the next one starts
    const std::vector<double> & returns() const
    {
        return returns_;
    }

    /// @brief start a new episode, closing the one before if it was cut short, and draw its first state
    /// @return the start state
    state_type reset()
    {
        if (episode_ > 0 && !closed_) {
            returns_.push_back(total_);
        }
        episode_++;
        steps_  = 0;
        total_  = 0;
        closed_ = false;
        at_ = game_.reset();
        frame();
        return at_;
    }

    /// @brief take action @param a in the wrapped game, and draw the state reached @return the outcome of the step
    step_result<state_type> step(const action_type & a)
    {
        auto out = game_.step(a);
        at_ = out.next;
        steps_++;
        total_ += out.reward;
        if (out.terminated) {
            // the episode ended: its return goes on the curve, and the last frame shows it
            returns_.push_back(total_);
            closed_ = true;
        }
        if (out.terminated || steps_ % std::max(every, 1u) == 0) {
            frame();
        }
        return out;
    }

    /// @return the legal actions of the wrapped game in state @param s
    std::vector<action_type> actions(const state_type & s) const
    {
        return game_.actions(s);
    }

    /// @brief draw the game as it is now, if `on`, and wait `delay` milliseconds; once the screen says it is
    ///        closed (a window shut by the user), nothing more is drawn
    void frame()
    {
        if (!on) {
            return;
        }
        if (!fitted_) {
            fit();
            fitted_ = true;
        }
        if (!screen_.show(drawing(), status(), title)) {
            on = false;
            return;
        }
        if (delay > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        }
    }

    /// @return the frame as a picture: the game, and the learning curve under it
    canvas drawing() const
    {
        canvas c(view_type::width(), view_type::height() + gap(), colours::black);
        view_type::draw(c, game_, at_, memory);
        if (curve_height > 0) {
            plot_curve(c, 0, static_cast<int>(view_type::height() + 1), view_type::width(), curve_height, returns_);
        }
        return c;
    }

    /// @return the frame as text, rendered for this terminal
    std::string picture() const
    {
        return terminal_screen::render(drawing());
    }

    /// @return the rows of a terminal a frame takes: the picture's lines and the status line
    unsigned int lines() const
    {
        return terminal_screen::lines_of(view_type::height() + gap());
    }

    /// @return the status line: the title, the episode, its steps and its return so far
    std::string status() const
    {
        std::ostringstream line;
        if (!title.empty()) {
            line << title << "  ";
        }
        line << "episode " << episode_ << "  step " << steps_ << "  return " << total_;
        return line.str();
    }

private:
    /// @return the pixels under the game: a blank row and the curve, if any
    unsigned int gap() const
    {
        return curve_height > 0 ? curve_height + 1 : 0;
    }

    /// @brief make the frame fit the screen, if it says how much fits: without the curve if need be, and with a
    ///        note if it cannot fit at all
    void fit()
    {
        if (!screen_.fits(lines())) {
            curve_height = 0;
        }
        if (!screen_.fits(lines())) {
            screen_.too_tall(lines());
        }
    }

    // the wrapped game
    game_type game_;
    // where the frames go
    screen_type screen_;
    // the state the game is in
    state_type at_{};
    // the episodes started, the steps and the return of the current one, whether it has ended, and the returns of
    // those before
    std::size_t episode_ = 0;
    std::size_t steps_ = 0;
    double total_ = 0;
    bool closed_ = false;
    std::vector<double> returns_;
    // true once the frame has been made to fit the screen
    bool fitted_ = false;
};

/**
 * @struct watch_options
 * @brief what the examples take from the command line: `--watch` to draw, `--delay ms` between frames, `--every n`
 *        steps, and, for a window, `--snapshot file.png` to save its last frame
 */
struct watch_options
{
    /// true with `--watch`
    bool on = false;
    /// milliseconds between frames (`--delay`), 30 unless given
    unsigned int delay = 30;
    /// steps between frames (`--every`), 1 unless given
    unsigned int every = 1;
    /// a file to save the window's last frame to (`--snapshot`), if the screen is a window
    std::string snapshot;

    /// @brief read the options from the command line
    static watch_options parse(int argc,
                               char ** argv)
    {
        watch_options options;
        for (int i = 1; i < argc; i++) {
            const std::string arg = argv[i];
            if (arg == "--watch") {
                options.on = true;
            }
            else if (arg == "--delay" && i + 1 < argc) {
                options.delay = static_cast<unsigned int>(std::max(0, std::atoi(argv[++i])));
            }
            else if (arg == "--every" && i + 1 < argc) {
                options.every = static_cast<unsigned int>(std::max(1, std::atoi(argv[++i])));
            }
            else if (arg == "--snapshot" && i + 1 < argc) {
                options.snapshot = argv[++i];
            }
        }
        return options;
    }

    /// @brief set @param game up as the options say, and its screen too: the terminal's height, @param rows (this
    ///        terminal's unless given), if it has rows, and the snapshot, if it takes one
    template <class game_type,
              class screen_type>
    void apply(watched<game_type, screen_type> & game,
               unsigned int rows = terminal::rows()) const
    {
        game.on    = on;
        game.delay = delay;
        game.every = every;
        set_rows(game.screen(), rows, 0);
        set_snapshot(game.screen(), snapshot, 0);
    }

private:
    /// @brief a screen with `rows` (the terminal) is told how many the terminal has
    template <class screen_type>
    static auto set_rows(screen_type & screen, unsigned int rows, int) -> decltype(screen.rows = rows, void())
    {
        screen.rows = rows;
    }

    /// @brief any other screen (a window) has no rows to be told
    template <class screen_type>
    static void set_rows(screen_type &, unsigned int, long)
    {}

    /// @brief a screen with a `snapshot` (a window) is told the file to save
    template <class screen_type>
    static auto set_snapshot(screen_type & screen, const std::string & file, int)
        -> decltype(screen.snapshot = file, void())
    {
        screen.snapshot = file;
    }

    /// @brief any other screen (the terminal) takes none
    template <class screen_type>
    static void set_snapshot(screen_type &, const std::string &, long)
    {}
};
} // namespace games
#endif
