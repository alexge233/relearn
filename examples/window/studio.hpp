#ifndef RELEARN_WINDOW_STUDIO_HPP
#define RELEARN_WINDOW_STUDIO_HPP
/**
 * Draws a lesson with raylib: the intro, the top bar, the formula with its real numbers, the plain words and the help.
 * What is written comes from lesson.hpp; a lesson's own board is a class with
 *
 *     void draw(studio & s, const session & now, const controls & shown, float x, float y) const;
 *     const char * legend() const;      // what the colours of the board mean
 *     float height() const;             // of the board, in screen pixels
 *     std::vector<std::string> moves() const;   // the names of the moves
 *     std::string describe(const session & now, int column, int row) const;   // what a bubble says about a cell
 *     int columns() const;  int rows() const;   // of the board
 *     float cell() const;               // the side of a cell, in screen pixels
 *
 * The music and the effects are chiptune.hpp's, played by `speaker`. Keys and mouse do the same things (H shows them all). The window is 1280 x 732, drawn at its native resolution.
 */
#include "chiptune.hpp"
#include "planning.hpp"
#include <raylib.h>
#include <rlgl.h>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef RELEARN_ASSETS
#define RELEARN_ASSETS "examples/window/assets"
#endif

namespace games
{
namespace lessons
{
/// @return the raylib colour of @param c, with alpha @param a
inline Color colour(const rgb & c,
                    unsigned char a = 255)
{
    return Color{c.r, c.g, c.b, a};
}

/// the palette of the lessons
namespace ink
{
const Color text{232, 232, 240, 255};
const Color dim{150, 156, 182, 255};
const Color top{16, 18, 34, 255};
const Color bottom{30, 26, 58, 255};
const Color panel{26, 28, 50, 255};
const Color line{58, 62, 96, 255};
const Color belief{110, 170, 255, 255};
const Color gold{255, 214, 90, 255};
const Color reward{120, 226, 140, 255};
const Color future{255, 130, 190, 255};
const Color next{255, 160, 80, 255};
const Color danger{240, 90, 80, 255};
}

/// @return the colour a term of the formula is written in
inline Color colour_of(role kind)
{
    switch (kind) {
        case role::result: return ink::text;
        case role::belief: return ink::belief;
        case role::speed:  return ink::gold;
        case role::reward: return ink::reward;
        case role::future: return ink::future;
        case role::next:   return ink::next;
        default:           return ink::dim;
    }
}

/**
 * @class studio
 * @brief the fonts, the tile sheets, and the shapes every lesson draws with
 */
class studio
{
public:
    /// screen pixels to a pixel of the art
    static constexpr int zoom = 3;
    /// the side of a tile, in the art's pixels
    static constexpr int tile_side = 16;
    /// the side of a tile on the screen
    static constexpr int cell = zoom * tile_side;

    /// @brief load the fonts and the tile sheets from the assets folder; the window must be open
    void load()
    {
        const std::string assets = RELEARN_ASSETS;
        regular_ = LoadFontEx((assets + "/PixelOperator.ttf").c_str(), 16, nullptr, 0);
        bold_    = LoadFontEx((assets + "/PixelOperator-Bold.ttf").c_str(), 16, nullptr, 0);
        town_    = LoadTexture((assets + "/kenney_tiny_town.png").c_str());
        dungeon_ = LoadTexture((assets + "/kenney_tiny_dungeon.png").c_str());
        SetTextureFilter(regular_.texture, TEXTURE_FILTER_POINT);
        SetTextureFilter(bold_.texture, TEXTURE_FILTER_POINT);
    }

    /// @brief free them, before the window closes
    void unload()
    {
        UnloadFont(regular_);
        UnloadFont(bold_);
        UnloadTexture(town_);
        UnloadTexture(dungeon_);
    }

    /// @brief write @param s at (@param x, @param y), at @param size pixels (a multiple of 16 stays sharp)
    void write(const std::string & s,
               float x,
               float y,
               int size,
               Color c,
               bool heavy = false) const
    {
        DrawTextEx(heavy ? bold_ : regular_, s.c_str(), {static_cast<float>(static_cast<int>(x)), static_cast<float>(static_cast<int>(y))},
                   static_cast<float>(size), 0, c);
    }

    /// @return the width of @param s at @param size
    float width(const std::string & s,
                int size,
                bool heavy = false) const
    {
        return MeasureTextEx(heavy ? bold_ : regular_, s.c_str(), static_cast<float>(size), 0).x;
    }

    /// @brief draw the tile in column @param col and row @param row of the town's sheet at (@param x, @param y)
    void town(int col,
              int row,
              float x,
              float y,
              int scale = zoom) const
    {
        tile(town_, col, row, x, y, WHITE, scale);
    }

    /// @brief draw a tile of the dungeon's sheet, tinted @param tint
    void dungeon(int col,
                 int row,
                 float x,
                 float y,
                 Color tint = WHITE,
                 int scale = zoom) const
    {
        tile(dungeon_, col, row, x, y, tint, scale);
    }

    /// @brief draw the skull at (@param x, @param y), the size of a tile
    void skull(float x,
               float y) const
    {
        const auto & rows = skull_art();
        for (std::size_t r = 0; r < rows.size(); r++) {
            for (std::size_t c = 0; c < rows[r].size(); c++) {
                if (rows[r][c] != '.') {
                    DrawRectangle(static_cast<int>(x) + static_cast<int>(c) * zoom, static_cast<int>(y) + static_cast<int>(r) * zoom, zoom, zoom, bone(rows[r][c]));
                }
            }
        }
    }

    /// @brief draw a panel with a title
    void panel(float x,
               float y,
               float w,
               float h,
               const std::string & title) const
    {
        DrawRectangleRounded({x, y, w, h}, 0.03f, 6, ink::panel);
        DrawRectangleRoundedLinesEx({x, y, w, h}, 0.03f, 6, 1, ink::line);
        if (!title.empty()) {
            write(title, x + 14, y + 10, 16, ink::gold, true);
        }
    }

    /// @brief draw an arrow for move @param a (up, right, down, left) centred on (@param cx, @param cy)
    void arrow(float cx,
               float cy,
               unsigned int a,
               Color c) const
    {
        static const float dirs[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
        const float dx = dirs[a][0];
        const float dy = dirs[a][1];
        const float s = 9;
        const Vector2 tip{cx + dx * s, cy + dy * s};
        const Vector2 one{cx - dx * s * 0.4f + dy * s * 0.7f, cy - dy * s * 0.4f - dx * s * 0.7f};
        const Vector2 two{cx - dx * s * 0.4f - dy * s * 0.7f, cy - dy * s * 0.4f + dx * s * 0.7f};
        DrawTriangle(tip, two, one, c);
        DrawTriangle(tip, one, two, c);
    }

    /**
     * @brief write a symbol of the formula, as a pixel glyph if it has one
     * @return its width
     */
    float symbol(const std::string & s,
                 float x,
                 float y,
                 Color c,
                 bool draw_it = true) const
    {
        const auto rows = glyph(s);
        if (rows.empty()) {
            if (draw_it) {
                write(s, x, y, 32, c, true);
            }
            return width(s, 32, true);
        }
        const int px = 2;
        if (draw_it) {
            for (std::size_t r = 0; r < rows.size(); r++) {
                for (std::size_t q = 0; q < rows[r].size(); q++) {
                    if (rows[r][q] == '#') {
                        DrawRectangle(static_cast<int>(x) + static_cast<int>(q) * px, static_cast<int>(y) + static_cast<int>(r) * px + 4, px, px, c);
                    }
                }
            }
        }
        return static_cast<float>(rows.front().size()) * px;
    }

private:
    /// @return the colour of a letter of the skull
    static Color bone(char letter)
    {
        switch (letter) {
            case 'o': return Color{236, 232, 220, 255};
            case 's': return Color{176, 172, 188, 255};
            default:  return Color{50, 34, 52, 255};
        }
    }

    /// @brief draw a tile of @param sheet, scaled by `zoom`
    void tile(const Texture2D & sheet,
              int col,
              int row,
              float x,
              float y,
              Color tint,
              int scale) const
    {
        const float side = static_cast<float>(tile_side);
        const float target = static_cast<float>(tile_side * scale);
        DrawTexturePro(sheet, {col * side, row * side, side, side}, {x, y, target, target}, {0, 0}, 0, tint);
    }

    Font regular_{};
    Font bold_{};
    Texture2D town_{};
    Texture2D dungeon_{};
};

/**
 * @class speaker
 * @brief plays the music and the effects; without a sound device, it does nothing
 */
class speaker
{
public:
    /// @brief open the sound device and start the music
    void open()
    {
        InitAudioDevice();
        if (!IsAudioDeviceReady()) {
            return;
        }
        SetAudioStreamBufferSizeDefault(chunk);
        music_ = LoadAudioStream(sample_rate, 16, 1);
        SetAudioStreamVolume(music_, 0.5f);
        fall_     = sound_of(book_.fall());
        treasure_ = sound_of(book_.treasure());
        click_    = sound_of(book_.click());
        PlayAudioStream(music_);
        open_ = true;
    }

    /// @brief free everything, before the window closes
    void close()
    {
        if (!open_) {
            return;
        }
        UnloadAudioStream(music_);
        UnloadSound(fall_);
        UnloadSound(treasure_);
        UnloadSound(click_);
        CloseAudioDevice();
        open_ = false;
    }

    /// @brief top up the music; once a frame
    void update()
    {
        if (open_ && IsAudioStreamProcessed(music_)) {
            short samples[chunk];
            cursor_.fill(samples, chunk);
            UpdateAudioStream(music_, samples, chunk);
        }
    }

    /// @brief turn everything on or off
    void set_on(bool on)
    {
        if (open_) {
            SetMasterVolume(on ? 1.0f : 0.0f);
        }
    }

    /// @brief play the sound of a step
    void play(cue what)
    {
        if (open_ && what != cue::none) {
            PlaySound(what == cue::fall ? fall_ : treasure_);
        }
    }

    /// @brief play the blip of a click
    void click()
    {
        if (open_) {
            PlaySound(click_);
        }
    }

private:
    /// samples handed to the music at a time
    static constexpr int chunk = 4096;

    /// @return @param samples as a raylib sound, which copies them
    static Sound sound_of(const std::vector<short> & samples)
    {
        Wave wave{};
        wave.frameCount = static_cast<unsigned int>(samples.size());
        wave.sampleRate = sample_rate;
        wave.sampleSize = 16;
        wave.channels   = 1;
        wave.data       = const_cast<short *>(samples.data());
        return LoadSoundFromWave(wave);
    }

    sound_book book_;
    loop_cursor cursor_{book_.music()};
    AudioStream music_{};
    Sound fall_{};
    Sound treasure_{};
    Sound click_{};
    bool open_ = false;
};

/**
 * @class lesson_app
 * @brief runs a lesson: the intro, then the main screen, until the window is closed
 * @tparam board_type draws the game
 */
template <class board_type>
class lesson_app
{
public:
    static constexpr int width_px  = 1280;
    static constexpr int height_px = 732;
    static constexpr int bar_height = 36;

    /**
     * @param about is what the lesson says about itself
     * @param learners are the learners it can switch between
     * @param board draws the game
     * @param options are the command line's
     */
    lesson_app(info about,
               std::vector<std::unique_ptr<session>> learners,
               board_type board,
               lesson_options options)
    : about_(std::move(about)),
      learners_(std::move(learners)),
      board_(std::move(board)),
      options_(std::move(options))
    {}

    /// @brief run the lesson @return the program's exit code
    int run()
    {
        const bool capture = !options_.capture.empty();
        SetTraceLogLevel(LOG_WARNING);
        SetConfigFlags(capture ? FLAG_WINDOW_HIDDEN : (FLAG_WINDOW_HIGHDPI | FLAG_VSYNC_HINT));
        InitWindow(width_px, height_px, ("relearn: " + about_.title).c_str());
        SetExitKey(KEY_NULL);
        SetTargetFPS(60);
        look_.load();
        prepare_crawl();
        in_intro_ = !options_.no_intro;
        shown_.sound = !options_.mute;
        if (!capture) {
            sound_.open();
            sound_.set_on(shown_.sound);
        }
        if (capture) {
            render_pictures();
        }
        else {
            while (!WindowShouldClose() && !quit_) {
                sound_.update();
                update(GetFrameTime());
                frame();
            }
        }
        sound_.close();
        UnloadRenderTexture(crawl_);
        look_.unload();
        CloseWindow();
        return 0;
    }

private:
    session & now()
    {
        return *learners_[current_];
    }

    /// @brief read the input, then advance the intro, or the game
    void update(double seconds)
    {
        if (in_intro_) {
            crawl_seconds_ += seconds;
            if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_ENTER) || IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                in_intro_ = false;
            }
            if (IsKeyPressed(KEY_Q) || IsKeyPressed(KEY_ESCAPE)) {
                quit_ = true;
            }
            return;
        }
        keyboard();
        drag_sliders();
        mouse();
        rest_ = seconds;
        shown_.death.advance(seconds);
        if (!shown_.help) {
            for (unsigned int n = shown_.steps_for(seconds); n > 0; n--) {
                now().step();
                if (shown_.followable_steps() && !imagining()) {
                    sound_.play(cue_of(now().punished(), now().latest().ended));
                    if (now().punished()) {
                        shown_.death.begin();
                        break;
                    }
                }
            }
        }
    }

    /// @brief turn the keys pressed into commands
    void keyboard()
    {
        static const int keys[] = {KEY_SPACE, KEY_N, KEY_RIGHT, KEY_R, KEY_Z, KEY_MINUS, KEY_EQUAL, KEY_KP_SUBTRACT, KEY_KP_ADD,
                                   KEY_V, KEY_A, KEY_I, KEY_M, KEY_H, KEY_Q, KEY_TAB};
        for (int key : keys) {
            if (IsKeyPressed(key)) {
                run_command(command_of_key(key));
            }
        }
        if (IsKeyPressed(KEY_S)) {
            TakeScreenshot(("relearn-lesson-" + std::to_string(++shots_) + ".png").c_str());
        }
        if (IsKeyPressed(KEY_ESCAPE)) {
            if (shown_.help) {
                shown_.help = false;
            }
            else if (picking_) {
                picking_ = false;
            }
            else {
                quit_ = true;
            }
        }
    }

    /// @return true if the last step was imagined
    bool imagining() const
    {
        const planner * p = learners_[current_]->plans();
        return p != nullptr && p->imagining();
    }

    /// @return how many sliders this learner has: a planner has a fourth
    int slider_count() const
    {
        return learners_[current_]->plans() != nullptr ? 4 : 3;
    }

    /// @return the values slider @param which goes through
    static slider_range range_of(int which)
    {
        return which == 3 ? slider_range{0, 50, 1} : slider_range{0, 1, 0.01};
    }

    /// @brief the settings, as sliders: alpha, gamma, epsilon, and n for a planner
    double setting(int which)
    {
        if (which == 3) {
            return now().plans()->planning_steps();
        }
        return which == 0 ? now().alpha() : (which == 1 ? now().gamma() : now().epsilon());
    }

    /// @brief set one of the three settings
    void set_setting(int which,
                     double value)
    {
        if (which == 3) {
            now().plans()->set_planning_steps(static_cast<unsigned int>(value + 0.5));
        }
        else if (which == 0) {
            now().set_alpha(value);
        }
        else if (which == 1) {
            now().set_gamma(value);
        }
        else {
            now().set_epsilon(value);
        }
    }

    /// @return the screen height of slider @param which
    float slider_y(int which) const
    {
        return slider_count() == 4 ? 556.0f + static_cast<float>(which) * 34.0f : 562.0f + static_cast<float>(which) * 42.0f;
    }

    /// @brief drag a slider with the mouse, and the button that puts the settings back
    void drag_sliders()
    {
        const Vector2 at = pointer();
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            dragging_ = -1;
        }
        if (shown_.help || picking_) {
            return;
        }
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            for (int i = 0; i < slider_count(); i++) {
                if (at.x >= slider_left - 8 && at.x <= slider_left + slider_width + 8 && at.y >= slider_y(i) - 8 && at.y <= slider_y(i) + 22) {
                    dragging_ = i;
                }
            }
            if (at.x >= 1096 && at.x <= 1240 && at.y >= 522 && at.y <= 548) {
                now().restore_settings();
            }
        }
        if (dragging_ >= 0) {
            set_setting(dragging_, slider_value(range_of(dragging_), at.x, slider_left, slider_width));
        }
    }

    /// @brief handle a click on the top bar or the learners' list
    void mouse()
    {
        if (!IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            return;
        }
        const Vector2 at = pointer();
        if (picking_) {
            const int row = static_cast<int>(at.y - bar_height) / picker_row;
            picking_ = false;
            if (at.y >= bar_height && at.x < picker_width && row >= 0 && row < static_cast<int>(learners_.size())) {
                current_ = static_cast<std::size_t>(row);
                shown_.death.stop();
                return;
            }
        }
        const command hit = menu_hit(menu(), at.x, at.y, bar_height);
        if (hit != command::none) {
            sound_.click();
        }
        if (hit == command::learner) {
            picking_ = true;
        }
        else if (shown_.help) {
            shown_.help = false;
        }
        else {
            run_command(hit);
        }
    }

    /// @brief do what a key or the top bar asked
    void run_command(command c)
    {
        switch (c) {
            case command::learner: current_ = (current_ + 1) % learners_.size(); shown_.death.stop(); break;
            case command::restart: now().restart(); shown_.death.stop(); break;
            case command::reset:   now().reset(); shown_.death.stop(); break;
            case command::pause:   shown_.paused = !shown_.paused; break;
            case command::step:    shown_.single_step(); break;
            case command::slower:  shown_.slower(); break;
            case command::faster:  shown_.faster(); break;
            case command::values:  shown_.values = !shown_.values; break;
            case command::arrows:  shown_.arrows = !shown_.arrows; break;
            case command::intro:   in_intro_ = true; crawl_seconds_ = 0; break;
            case command::sound:   shown_.sound = !shown_.sound; sound_.set_on(shown_.sound); break;
            case command::help:    shown_.help = !shown_.help; break;
            case command::quit:    quit_ = true; break;
            default: break;
        }
    }

    /// @return the top bar as it is now
    std::vector<menu_item> menu() const
    {
        const std::vector<std::pair<command, std::string>> entries = {
            {command::learner, learners_[current_]->name() + " v"}, {command::restart, "Restart"}, {command::reset, "Reset"},
            {command::pause, shown_.paused ? "Resume" : "Pause"}, {command::step, "Step"}, {command::slower, "Slower"},
            {command::faster, "Faster"}, {command::values, "Values"}, {command::arrows, "Arrows"}, {command::intro, "Intro"},
            {command::sound, shown_.sound ? "Sound on" : "Sound off"}, {command::help, "Help"}};
        return lay_out_menu(entries, [this](const std::string & s) { return look_.width(s, 16); }, 8, 12);
    }

    /// @return where the mouse is (or where the pictures pretend it is)
    Vector2 pointer() const
    {
        return pretending_ ? pretend_ : GetMousePosition();
    }

    /// @return the columns of the formula, and its terms' widths in @param widths
    std::vector<column> layout_formula(const std::vector<term> & terms,
                                       std::vector<float> & widths) const
    {
        widths.clear();
        for (const auto & t : terms) {
            widths.push_back(std::max({look_.symbol(t.symbol, 0, 0, ink::text, false), look_.width(t.number, 16, true), look_.width(t.label, 16)}));
        }
        return formula_columns(widths, 36, width_px - 72, 24);
    }

    /// @return every part of the main screen that explains itself; later ones are on top
    std::vector<hot_zone> zones()
    {
        std::vector<hot_zone> out;
        out.push_back({32, 46, 576, 40, "The demo's name, which learner is running, and how the current episode is going. The return is the sum of its rewards so far."});
        out.push_back({32, 92 + board_.height(), 576, 32, "The colour scale: blue is where it expects to do badly, gold where it expects to do well."});
        out.push_back({640, 50, 304, 34, "The value of each move from the cell it just left. It usually picks the highest. The white outline is the one it took."});
        out.push_back({640, 278, 304, 40, "Epsilon picks the move, before the formula runs, so Q-learning's formula has no epsilon: its max ignores what happens next. SARSA feels it through the move it really makes, and Expected SARSA averages with it."});
        out.push_back({956, 50, 300, 270, now().curve_help()});
        out.push_back({24, 332, static_cast<float>(width_px) - 48, 36, "The whole learning rule. Rest the mouse on any term to see what it means."});
        out.push_back({24, 516, 776, 204, "The step that just happened, read out as sentences. It is the formula above, in words."});
        out.push_back({812, 516, 444, 40, "Drag a slider and the learner changes from its next step. Each learner keeps its own settings."});
        out.push_back({820, slider_y(0) - 10, 430, 34, "alpha, the learning rate. High: learns fast but jumps around. Low: slow and steady. At 0 it never learns."});
        out.push_back({820, slider_y(1) - 10, 430, 34, "gamma, the discount. Lower it and the walker cares less about the far-off treasure: values fade with distance. At 1 all rewards count."});
        out.push_back({820, slider_y(2) - 10, 430, 34, "epsilon, how often it moves at random. At 0 it never tries anything new; at 1 it ignores what it has learnt. Watch how many falls change."});
        if (slider_count() == 4) {
            out.push_back({820, slider_y(3) - 10, 430, 34, "n, how many steps it imagines after each real one. At 0 it is plain Q-learning. Try 50: it needs far fewer real moves."});
        }
        const auto names = board_.moves();
        const position p = now().from();
        for (std::size_t a = 0; a < names.size(); a++) {
            const Vector2 middle = wedge_middle(static_cast<unsigned int>(a));
            out.push_back({middle.x - 44, middle.y - 30, 88, 60, "Q(s, " + names[a] + "): what it believes moving " + names[a] + " from that cell is worth: " + two_decimals(now().q(p, static_cast<unsigned int>(a))) + ". Gold is the best of the four, blue the worst."});
        }
        const float side = board_.cell();
        for (int r = 0; r < board_.rows(); r++) {
            for (int c = 0; c < board_.columns(); c++) {
                out.push_back({32 + c * side, 92 + r * side, side, side, board_.describe(now(), c, r)});
            }
        }
        const auto terms = formula_terms(now().latest(), now().next_symbol(), now().next_label());
        std::vector<float> widths;
        const auto columns = layout_formula(terms, widths);
        for (std::size_t i = 0; i < terms.size(); i++) {
            const std::string text = term_help(terms[i], now().idea());
            if (!text.empty()) {
                out.push_back({columns[i].x, 368, columns[i].width, 134, text});
            }
        }
        for (const auto & item : menu()) {
            out.push_back({item.x, 0, item.width, static_cast<float>(bar_height), menu_help(item.what)});
        }
        return out;
    }

    /// @brief the help bubble of what the mouse has rested on
    void draw_bubble()
    {
        const Vector2 at = pointer();
        const auto all = zones();
        const hot_zone * zone = (picking_ || shown_.help) ? nullptr : zone_at(all, at.x, at.y);
        // the zone's place identifies it: its text changes with every step, and the bubble must not start over
        if (!hover_.update(zone ? std::to_string(zone->x) + "," + std::to_string(zone->y) : std::string(), rest_)) {
            return;
        }
        const auto lines = wrap_text(zone->text, 46);
        float wide = 0;
        for (const auto & line : lines) {
            wide = std::max(wide, look_.width(line, 16));
        }
        const float pad = 10;
        const float w = wide + 2 * pad;
        const float h = static_cast<float>(lines.size()) * 20 + 2 * pad - 2;
        const auto place = place_bubble(at.x, at.y, w, h, width_px, height_px);
        DrawRectangleRounded({place.x + 3, place.y + 3, w, h}, 0.15f, 6, Color{0, 0, 0, 120});
        DrawRectangleRounded({place.x, place.y, w, h}, 0.15f, 6, Color{58, 54, 104, 250});
        DrawRectangleRoundedLinesEx({place.x, place.y, w, h}, 0.15f, 6, 1, ink::gold);
        float y = place.y + pad;
        for (const auto & line : lines) {
            look_.write(line, place.x + pad, y, 16, ink::text);
            y += 20;
        }
    }

    /// @brief draw one frame
    void frame()
    {
        BeginDrawing();
        ClearBackground(BLACK);
        if (in_intro_) {
            draw_intro();
        }
        else {
            draw_main();
        }
        EndDrawing();
    }

    /// @brief write the intro on a texture, once, for the crawl to scroll
    void prepare_crawl()
    {
        std::vector<std::pair<std::string, int>> lines;
        lines.push_back({about_.title, 64});
        lines.push_back({about_.source, 32});
        lines.push_back({"", 32});
        for (const auto & paragraph : about_.intro) {
            for (const auto & line : wrap_text(paragraph, 44)) {
                lines.push_back({line, 32});
            }
            lines.push_back({"", 32});
        }
        crawl_height_ = 0;
        for (const auto & line : lines) {
            crawl_height_ += line.second + 14;
        }
        crawl_ = LoadRenderTexture(crawl_width, crawl_height_ + 40);
        SetTextureFilter(crawl_.texture, TEXTURE_FILTER_BILINEAR);
        SetTextureWrap(crawl_.texture, TEXTURE_WRAP_CLAMP);
        BeginTextureMode(crawl_);
        ClearBackground(BLANK);
        float y = 20;
        for (const auto & line : lines) {
            look_.write(line.first, (crawl_width - look_.width(line.first, line.second, true)) / 2, y, line.second, ink::gold, true);
            y += static_cast<float>(line.second) + 14;
        }
        EndTextureMode();
    }

    /// @brief scatter the stars
    void draw_stars() const
    {
        SetRandomSeed(7);
        for (int i = 0; i < 420; i++) {
            const int x = GetRandomValue(0, width_px);
            const int y = GetRandomValue(0, height_px);
            const auto b = static_cast<unsigned char>(GetRandomValue(70, 255));
            DrawPixel(x, y, Color{b, b, static_cast<unsigned char>(std::min(255, b + 20)), 255});
            if (i % 23 == 0) {
                DrawRectangle(x, y, 2, 2, Color{255, 255, 255, b});
            }
        }
    }

    /// @brief the intro: the text crawling away over the stars, tilted little enough to read
    void draw_intro()
    {
        DrawRectangleGradientV(0, 0, width_px, height_px, Color{4, 4, 14, 255}, Color{16, 10, 38, 255});
        draw_stars();
        // the camera looks down on the plane of the text at about 55 degrees: flatter than that is hard to read
        Camera3D camera{};
        camera.position = {0.0f, 8.6f, 5.2f};
        camera.target = {0.0f, 0.0f, -0.9f};
        camera.up = {0.0f, 1.0f, 0.0f};
        camera.fovy = 45.0f;
        camera.projection = CAMERA_PERSPECTIVE;
        const float half = 6.0f;
        const float near_z = 4.2f;
        const float far_z = -11.0f;
        const float pixels_per_unit = crawl_width / (2 * half);
        const float view = (near_z - far_z) * pixels_per_unit;
        const float height = static_cast<float>(crawl_.texture.height);
        const float scrolled = static_cast<float>(crawl_position(crawl_seconds_, crawl_speed));
        // the texture is upside down; the near edge of the plane shows the row `scrolled`, the far edge the row before
        const float near_v = 1.0f - scrolled / height;
        const float far_v = 1.0f - (scrolled - view) / height;
        BeginMode3D(camera);
        rlSetTexture(crawl_.texture.id);
        rlBegin(RL_QUADS);
        rlColor4ub(255, 255, 255, 255);
        rlTexCoord2f(0, far_v);
        rlVertex3f(-half, 0, far_z);
        rlTexCoord2f(0, near_v);
        rlVertex3f(-half, 0, near_z);
        rlTexCoord2f(1, near_v);
        rlVertex3f(half, 0, near_z);
        rlTexCoord2f(1, far_v);
        rlVertex3f(half, 0, far_z);
        rlEnd();
        rlSetTexture(0);
        EndMode3D();
        DrawRectangleGradientV(0, 0, width_px, 170, Color{4, 4, 14, 255}, Color{4, 4, 14, 0});
        const std::string prompt = "[ press space to continue ]";
        look_.write(prompt, (width_px - look_.width(prompt, 16)) / 2, height_px - 40, 16, ink::text);
    }

    /// @brief the top bar
    void draw_bar()
    {
        DrawRectangleGradientV(0, 0, width_px, bar_height, Color{44, 46, 84, 255}, Color{30, 32, 60, 255});
        DrawLine(0, bar_height, width_px, bar_height, ink::line);
        const Vector2 at = pointer();
        for (const auto & item : menu()) {
            const bool hover = at.y < bar_height && at.x >= item.x && at.x < item.x + item.width;
            const bool lit = (item.what == command::values && shown_.values) || (item.what == command::arrows && shown_.arrows) ||
                             (item.what == command::learner);
            if (hover) {
                DrawRectangleRounded({item.x + 2, 5, item.width - 4, 26}, 0.3f, 6, Color{70, 74, 130, 255});
            }
            look_.write(item.label, item.x + 12, 10, 16, lit ? ink::gold : ink::text);
        }
        const std::string rate = two_decimals(shown_.rate()).substr(0, two_decimals(shown_.rate()).find('.')) + " steps/s";
        look_.write(rate, width_px - 14 - look_.width(rate, 16), 10, 16, ink::dim);
    }

    /// @brief the learners' list, under the top bar
    void draw_picker()
    {
        DrawRectangle(0, bar_height, picker_width, picker_row * static_cast<int>(learners_.size()), Color{44, 46, 84, 255});
        DrawRectangleLines(0, bar_height, picker_width, picker_row * static_cast<int>(learners_.size()), ink::line);
        for (std::size_t i = 0; i < learners_.size(); i++) {
            look_.write(learners_[i]->name(), 16, bar_height + 6 + static_cast<float>(i) * picker_row, 16, i == current_ ? ink::gold : ink::text);
        }
    }

    /// @brief what the agent believes about the moves of the cell it just left
    void draw_beliefs()
    {
        look_.panel(640, 50, 304, 270, "Its beliefs in that cell");
        const auto names = board_.moves();
        const position p = now().from();
        double low = 0;
        double high = 0;
        for (std::size_t a = 0; a < names.size(); a++) {
            const double v = now().q(p, static_cast<unsigned int>(a));
            low = a == 0 ? v : std::min(low, v);
            high = a == 0 ? v : std::max(high, v);
        }
        const bool moved = now().latest().valid;
        for (unsigned int a = 0; a < names.size(); a++) {
            draw_wedge(a, colour(heat_colour(belief_heat(now().q(p, a), low, high))));
        }
        if (moved) {
            draw_wedge_outline(now().last_action());
        }
        for (unsigned int a = 0; a < names.size(); a++) {
            write_on_heat(names[a], a, -18, moved && a == now().last_action());
            write_on_heat(two_decimals(now().q(p, a)), a, 0, moved && a == now().last_action());
        }
        DrawCircle(static_cast<int>(wedges_x), static_cast<int>(wedges_y), 15, ink::panel);
        const float side = look_.width("s", 16, true);
        look_.write("s", wedges_x - side / 2, wedges_y - 9, 16, ink::text, true);
        if (imagining()) {
            look_.write("Imagined " + names[now().last_action()] + ": from memory", 654, 280, 16, ink::belief, true);
        }
        else if (moved) {
            look_.write(choice_words(names[now().last_action()], now().explored()), 654, 280, 16, now().explored() ? ink::next : ink::gold, true);
        }
        look_.write(exploring_words(now().epsilon()), 654, 298, 16, ink::dim);
    }

    /// @return which corners are the outer ends of the wedge of move @param a (up, right, down, left)
    static std::pair<int, int> wedge_corners(unsigned int a)
    {
        return {static_cast<int>(a), static_cast<int>((a + 1) % 4)};
    }

    /// @return a corner of the square of beliefs, clockwise from the top left
    static Vector2 corner(int which)
    {
        static const float signs[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        return {wedges_x + signs[which][0] * wedges_half, wedges_y + signs[which][1] * wedges_half};
    }

    /// @brief fill the wedge of move @param a
    void draw_wedge(unsigned int a,
                    Color c) const
    {
        const auto ends = wedge_corners(a);
        const Vector2 centre{wedges_x, wedges_y};
        DrawTriangle(centre, corner(ends.first), corner(ends.second), c);
        DrawTriangle(centre, corner(ends.second), corner(ends.first), c);
        DrawLineEx(centre, corner(ends.first), 2, ink::panel);
        DrawLineEx(centre, corner(ends.second), 2, ink::panel);
    }

    /// @brief ring the wedge of the move it took
    void draw_wedge_outline(unsigned int a) const
    {
        const auto ends = wedge_corners(a);
        const Vector2 centre{wedges_x, wedges_y};
        DrawLineEx(centre, corner(ends.first), 3, WHITE);
        DrawLineEx(corner(ends.first), corner(ends.second), 3, WHITE);
        DrawLineEx(corner(ends.second), centre, 3, WHITE);
    }

    /// @return the middle of the wedge of move @param a
    static Vector2 wedge_middle(unsigned int a)
    {
        static const float dirs[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
        return {wedges_x + dirs[a][0] * wedges_half * 0.6f, wedges_y + dirs[a][1] * wedges_half * 0.6f};
    }

    /// @brief write @param text in the wedge of move @param a, @param dy below its middle, with a shadow to read on any colour
    void write_on_heat(const std::string & text,
                       unsigned int a,
                       float dy,
                       bool heavy) const
    {
        const Vector2 at = wedge_middle(a);
        const float x = at.x - look_.width(text, 16, heavy) / 2;
        const float y = at.y - 1 + dy;
        look_.write(text, x + 1, y + 1, 16, Color{0, 0, 0, 220}, heavy);
        look_.write(text, x, y, 16, WHITE, heavy);
    }

    /// @brief the learning curve
    void draw_curve()
    {
        look_.panel(956, 50, 300, 270, "Learning curve");
        const float x = 970;
        const float y = 86;
        const float w = 272;
        const float h = 200;
        DrawLine(static_cast<int>(x), static_cast<int>(y + h), static_cast<int>(x + w), static_cast<int>(y + h), ink::line);
        DrawLine(static_cast<int>(x), static_cast<int>(y), static_cast<int>(x), static_cast<int>(y + h), ink::line);
        look_.write(now().curve_label(), x, y + h + 6, 16, ink::dim);
        auto values = curve_points(live_curve(now()), static_cast<std::size_t>(w) / 2);
        if (values.empty()) {
            return;
        }
        if (values.size() == 1) {
            // one episode so far: a level line, at what it has earned
            values.push_back(values.front());
        }
        const double low = lowest_shown(values, -300.0);
        double high = low;
        for (auto v : values) {
            high = std::max(high, v);
        }
        const auto row = [&](double v) {
            return y + h * static_cast<float>(1 - scaled(std::max(v, low), low, high));
        };
        const auto count = static_cast<float>(values.size() - 1);
        for (std::size_t i = 0; i + 1 < values.size(); i++) {
            const float x1 = x + w * static_cast<float>(i) / count;
            const float x2 = x + w * static_cast<float>(i + 1) / count;
            const float y1 = row(values[i]);
            const float y2 = row(values[i + 1]);
            // the shading under the line, one column of pixels at a time, so that it follows the slope
            for (int column = static_cast<int>(x1); column < static_cast<int>(x2) + 1; column++) {
                const float along = x2 > x1 ? std::max(0.0f, std::min(1.0f, (static_cast<float>(column) - x1) / (x2 - x1))) : 0.0f;
                const float surface = y1 + (y2 - y1) * along;
                DrawRectangleGradientV(column, static_cast<int>(surface), 1, static_cast<int>(y + h - surface),
                                       Fade(ink::reward, 0.26f), Fade(ink::reward, 0.02f));
            }
            DrawLineEx({x1, y1}, {x2, y2}, 1.5f, ink::reward);
        }
        const std::string top = plain_number(high);
        look_.write(top, x + 6, y, 16, ink::dim);
    }

    /// @brief the formula, each term over its number and what it means
    void draw_formula()
    {
        const planner * p = learners_[current_]->plans();
        look_.panel(24, 332, width_px - 48, 176, formula_title(imagining(), p != nullptr ? p->imagined_count() : 0, p != nullptr ? p->planning_steps() : 0));
        const auto terms = formula_terms(now().latest(), now().next_symbol(), now().next_label());
        std::vector<float> widths;
        const auto columns = layout_formula(terms, widths);
        const float y = 372;
        for (std::size_t i = 0; i < terms.size(); i++) {
            const term & t = terms[i];
            const Color c = colour_of(t.kind);
            const float centre = columns[i].x + columns[i].width / 2;
            const float wide = look_.symbol(t.symbol, 0, 0, c, false);
            look_.symbol(t.symbol, centre - wide / 2, y, c);
            look_.write(t.number, centre - look_.width(t.number, 16, true) / 2, y + 52, 16, c, true);
            if (!t.label.empty()) {
                DrawRectangle(static_cast<int>(centre - widths[i] / 2), static_cast<int>(y) + 42, static_cast<int>(widths[i]), 2, Fade(c, 0.7f));
                look_.write(t.label, centre - look_.width(t.label, 16) / 2, y + 80, 16, c);
            }
        }
    }

    /// @brief the same step in plain words, and what this learner does differently
    void draw_words()
    {
        look_.panel(24, 516, 776, 204, "In plain words");
        float y = 544;
        if (now().latest().valid) {
            const auto names = board_.moves();
            const auto & words = imagining() ? imagined_words : plain_words;
            for (const auto & line : words(now().latest(), names[now().last_action()], now().next_words())) {
                look_.write(line, 40, y, 16, ink::text);
                y += 22;
            }
        }
        else {
            look_.write("Nothing has happened yet: the first step will appear here.", 40, y, 16, ink::text);
            y += 88;
        }
        y += 8;
        look_.write(about_.watch_for, 40, y, 16, ink::reward);
        look_.write(now().idea(), 40, y + 20, 16, ink::gold);
        const std::string policy = "It follows the arrows, with a random move " + std::to_string(static_cast<int>(now().epsilon() * 100 + 0.5)) + "% of the time, so that it keeps exploring.";
        look_.write(policy, 40, y + 40, 16, ink::dim);
    }

    /// @brief the settings: drag them whilst it runs
    void draw_settings()
    {
        look_.panel(812, 516, 444, 204, "Try other values");
        DrawRectangleRounded({1096, 522, 144, 26}, 0.3f, 6, Color{58, 54, 104, 255});
        look_.write("Lesson's values", 1110, 527, 16, ink::text);
        const char * glyphs[4] = {"alpha", "gamma", "epsilon", "n"};
        const char * words[4] = {"speed", "future", "explore", "imagine"};
        const Color colours[4] = {ink::gold, ink::future, ink::belief, ink::next};
        for (int i = 0; i < slider_count(); i++) {
            const float y = slider_y(i);
            const slider_range range = range_of(i);
            look_.write(words[i], 828, y, 16, colours[i]);
            look_.symbol(glyphs[i], 894, y - 12, colours[i]);
            DrawRectangleRounded({slider_left, y + 4, slider_width, 8}, 0.5f, 6, Color{44, 46, 80, 255});
            const float at = slider_left + slider_width * static_cast<float>(slider_fraction(range, setting(i)));
            DrawRectangleRounded({slider_left, y + 4, at - slider_left, 8}, 0.5f, 6, colours[i]);
            DrawCircle(static_cast<int>(at), static_cast<int>(y) + 8, 8, dragging_ == i ? WHITE : ink::text);
            const std::string number = i == 3 ? std::to_string(static_cast<int>(setting(i))) : two_decimals(setting(i));
            look_.write(number, 1242 - look_.width(number, 16), y, 16, colours[i], true);
        }
        look_.write("Changes apply from the next step. Z starts afresh.", 828, 694, 16, ink::dim);
    }

    /// @brief the help, over everything
    void draw_help()
    {
        DrawRectangle(0, 0, width_px, height_px, Color{4, 4, 14, 215});
        look_.panel(170, 22, 940, 688, "Help");
        struct row { const char * key; const char * what; };
        static const row keys[] = {{"Space", "pause and resume"}, {"N or Right", "one step"}, {"- and =", "slower and faster"},
                                   {"R", "restart: a new episode, keeping what was learnt"}, {"Z", "reset: forget everything"},
                                   {"Tab", "the next learner (each one learns on its own)"}, {"V and A", "the colours and the arrows, on or off"},
                                   {"I", "the intro again"}, {"M", "the music and the sounds, on or off"}, {"S", "save a screenshot"}, {"H", "this help"}, {"Q", "quit"}};
        float y = 54;
        for (const auto & k : keys) {
            look_.write(k.key, 196, y, 16, ink::gold, true);
            look_.write(k.what, 340, y, 16, ink::text);
            y += 24;
        }
        y += 12;
        look_.write("What you are looking at", 196, y, 16, ink::gold, true);
        y += 28;
        for (const std::string & s : {std::string("The board: the game. ") + board_.legend(),
                                      std::string("Its beliefs: Q(s,a), what it thinks each move from the last cell is worth, gold for the best, blue for the worst. The white outline is the move it took."),
                                      std::string("Learning curve: the total reward of each episode. Higher is better."),
                                      std::string("The formula: the whole learning rule. Every term has its real value at this step."),
                                      std::string("In plain words: the same step, as a sentence."),
                                      std::string("alpha (speed): how far a belief moves towards what really happened."),
                                      std::string("gamma (future): how much what comes later counts against what comes now."),
                                      std::string("epsilon (explore): how often it moves at random instead of following the arrows. It picks the move, so it is not a term of Q-learning's formula. Drag the three sliders to try other values.")}) {
            for (const auto & line : wrap_text(s, 112)) {
                look_.write(line, 196, y, 16, ink::text);
                y += 22;
            }
            y += 4;
        }
        look_.write("Mouse: every entry of the top bar can be clicked. Click anywhere, or press H, to close this.", 196, 682, 16, ink::dim);
    }

    /// @brief the main screen
    void draw_main()
    {
        DrawRectangleGradientV(0, 0, width_px, height_px, ink::top, ink::bottom);
        look_.write(about_.title, 32, 46, 32, ink::text, true);
        const std::string state = now().name() + "   episode " + std::to_string(now().episode()) + "   step " + std::to_string(now().steps()) +
                                  "   " + now().tally() + (shown_.paused ? "   PAUSED" : "");
        look_.write(state, 44 + look_.width(about_.title, 32, true), 60, 16, shown_.paused ? ink::gold : ink::dim);
        board_.draw(look_, now(), shown_, 32, 92);
        const float scale_y = 92 + board_.height() + 8;
        for (int i = 0; i < 576; i++) {
            DrawRectangle(32 + i, static_cast<int>(scale_y), 1, 8, colour(heat_colour(i / 575.0)));
        }
        look_.write("worse", 32, scale_y + 8, 16, ink::dim);
        const std::string better = "better";
        look_.write(better, 32 + 576 - look_.width(better, 16), scale_y + 8, 16, ink::dim);
        const std::string meaning = "how good it believes each cell is";
        look_.write(meaning, 32 + 288 - look_.width(meaning, 16) / 2, scale_y + 8, 16, ink::text);
        draw_beliefs();
        draw_curve();
        draw_formula();
        draw_words();
        draw_settings();
        draw_bar();
        if (picking_) {
            draw_picker();
        }
        if (shown_.help) {
            draw_help();
        }
        draw_bubble();
    }

    /// @brief render the intro, the main screen and the help to files, without showing a window
    void render_pictures()
    {
        const std::string folder = options_.capture + "/";
        crawl_seconds_ = 16;
        in_intro_ = true;
        frame();
        frame();
        TakeScreenshot((folder + "intro.png").c_str());
        in_intro_ = false;
        // stop at an update with something to explain: the first, after some learning, with a real surprise
        for (int i = 0; i < 20000; i++) {
            now().step();
            if (i > 900 && std::fabs(now().latest().surprise()) > 2 && !now().punished()) {
                break;
            }
        }
        shown_.paused = true;
        frame();
        frame();
        TakeScreenshot((folder + "main.png").c_str());
        // the walker's death, half way through the skull and half way through the ghost
        for (int i = 0; i < 200000 && !now().punished(); i++) {
            now().step();
        }
        shown_.death.begin();
        for (const double seconds : {0.15, 0.27, 0.3}) {
            if (!now().punished()) {
                break;
            }
            shown_.death.advance(seconds);
            frame();
            frame();
            TakeScreenshot((folder + "death" + std::to_string(static_cast<int>(shown_.death.shown())) + ".png").c_str());
        }
        shown_.death.stop();
        // a bubble, over the discount of the formula
        pretending_ = true;
        pretend_ = {708, 400};
        rest_ = hover_timer::wait;
        frame();
        frame();
        frame();
        TakeScreenshot((folder + "bubble.png").c_str());
        pretending_ = false;
        shown_.help = true;
        frame();
        frame();
        TakeScreenshot((folder + "help.png").c_str());
    }

    static constexpr int crawl_width = 960;
    static constexpr double crawl_speed = 46;
    static constexpr int picker_width = 220;
    static constexpr int picker_row = 28;
    static constexpr float slider_left = 930;
    static constexpr float slider_width = 260;

    info about_;
    std::vector<std::unique_ptr<session>> learners_;
    board_type board_;
    lesson_options options_;
    /// the square of beliefs: its centre and half its side
    static constexpr float wedges_x = 792;
    static constexpr float wedges_y = 180;
    static constexpr float wedges_half = 92;

    studio look_;
    speaker sound_;
    controls shown_;
    std::size_t current_ = 0;
    bool in_intro_ = true;
    bool picking_ = false;
    bool quit_ = false;
    unsigned int shots_ = 0;
    double crawl_seconds_ = 0;
    hover_timer hover_;
    double rest_ = 0;
    bool pretending_ = false;
    int dragging_ = -1;
    Vector2 pretend_{};
    int crawl_height_ = 0;
    RenderTexture2D crawl_{};
};
} // namespace lessons
} // namespace games
#endif
