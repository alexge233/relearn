#ifndef RELEARN_WINDOW_WINDOW_HPP
#define RELEARN_WINDOW_WINDOW_HPP
/**
 * Watch the agents play, in a window: the same frames as the terminal's (have a look at games/watch.hpp), drawn
 * with raylib as retro pixels, every canvas pixel a sharp square, and the status line under them.
 *
 *     games::watched<games::cliff_walking, games::window_screen> game{games::cliff_walking()};
 *     games::play_online(game, agent, learner, memory, 1000);
 *
 * The examples are built with it as `win_<game>` by the optional sub-project (`cmake -DBUILD_WINDOW=On`), and
 * then `--watch` opens a window instead of drawing in the terminal. Whilst it is open:
 * - space pauses and resumes
 * - S saves a screenshot, `relearn-<n>.png`, in the current directory
 * - closing the window (or Escape) stops the watching for the rest of the program, which goes on without it
 * - when its game is over the window stays open on the last frame, until space is pressed or it is closed; with a
 *   `snapshot` to save, nobody is assumed to be watching, and it closes by itself
 *
 * The window is only looked after whilst a frame is being shown: between two watched episodes the program is busy
 * learning, and the window sits still (the system may call it unresponsive) until the next one.
 *
 * raylib is a C library, so this header is the only one that includes it, and nothing in `relearn.hpp` or in the
 * default build depends on it.
 */
#include "layout.hpp"
#include <raylib.h>
#include <string>
#include <utility>
#include <vector>

namespace games
{
/**
 * @class window_screen
 * @brief a screen for `watched` that shows the frames in a raylib window, opened at the first frame
 */
class window_screen
{
public:
    /// a screenshot to save when the window closes, if not empty: the last frame shown
    std::string snapshot;

    /// @brief a window showing @param pixel screen pixels to a canvas pixel (0: as many as fit the screen)
    explicit window_screen(unsigned int pixel = 0)
    : pixel_(pixel)
    {}

    // a window is not copied: `watched` moves it in
    window_screen(const window_screen &) = delete;
    window_screen & operator=(const window_screen &) = delete;
    window_screen(window_screen && other)
    : snapshot(std::move(other.snapshot)),
      pixel_(other.pixel_)
    {}

    /// @brief close the window: at once, saving the screenshot, if one was asked for; else once the user has seen it
    ~window_screen()
    {
        if (open_) {
            if (!snapshot.empty()) {
                TakeScreenshot(snapshot.c_str());
            }
            else {
                hold();
            }
            UnloadTexture(texture_);
            CloseWindow();
        }
    }

    /// @return true: a window always fits, whatever the frame's rows
    bool fits(unsigned int) const
    {
        return true;
    }

    /// @brief nothing to say: a window always fits
    void too_tall(unsigned int)
    {}

    /**
     * @brief show one frame, opening the window on the first with @param title in its bar
     * @return false once the user has closed the window: nothing is shown any more
     */
    bool show(const canvas & c,
              const std::string & status,
              const std::string & title)
    {
        if (dismissed()) {
            return false;
        }
        if (!open_) {
            start(c, title);
        }
        pixels_ = window_layout::bytes(c);
        UpdateTexture(texture_, pixels_.data());
        status_ = status;
        do {
            if (WindowShouldClose()) {
                dismissed() = true;
                return false;
            }
            if (IsKeyPressed(KEY_SPACE)) {
                paused_ = !paused_;
            }
            if (IsKeyPressed(KEY_S)) {
                TakeScreenshot(("relearn-" + std::to_string(++shots_) + ".png").c_str());
            }
            draw();
            if (paused_) {
                // nothing changes whilst paused: draw at about 60 frames a second, not as fast as the machine can
                WaitTime(1.0 / 60.0);
            }
        } while (paused_);
        return true;
    }

    /// @return true whilst the window is open
    bool open() const
    {
        return open_ && !dismissed();
    }

private:
    /// @return true once the user has closed a window: an example watches one game after another, each with its own
    ///         window, and a window closed by the user means no more of them
    static bool & dismissed()
    {
        static bool closed = false;
        return closed;
    }

    /// @brief open the window for a canvas the size of @param c, and the texture its pixels go to
    void start(const canvas & c,
               const std::string & title)
    {
        layout_ = window_layout(c.width(), c.height(), pixel_);
        const std::string bar = title.empty() ? "relearn" : "relearn: " + title;
        SetTraceLogLevel(LOG_WARNING);
        InitWindow(static_cast<int>(layout_.window_width()), static_cast<int>(layout_.window_height()), bar.c_str());
        // the pixels as a texture, drawn scaled up with no smoothing, so that every one stays a sharp square
        pixels_ = window_layout::bytes(c);
        Image image;
        image.data    = pixels_.data();
        image.width   = static_cast<int>(c.width());
        image.height  = static_cast<int>(c.height());
        image.mipmaps = 1;
        image.format  = PIXELFORMAT_UNCOMPRESSED_R8G8B8;
        texture_ = LoadTextureFromImage(image);
        SetTextureFilter(texture_, TEXTURE_FILTER_POINT);
        open_ = true;
    }

    /// @brief keep the window on its last frame until space is pressed or it is closed; a window the user has
    ///        already closed is not held
    void hold()
    {
        if (dismissed()) {
            return;
        }
        paused_ = false;
        status_ += "  finished: press space";
        while (!IsKeyPressed(KEY_SPACE)) {
            if (WindowShouldClose()) {
                dismissed() = true;
                return;
            }
            draw();
            WaitTime(1.0 / 60.0);
        }
    }

    /// @brief draw the texture and the status line
    void draw() const
    {
        BeginDrawing();
        ClearBackground(Color{0, 0, 0, 255});
        DrawTextureEx(texture_, Vector2{static_cast<float>(layout_.picture_left()), 0.0f}, 0.0f,
                      static_cast<float>(layout_.pixel()), Color{255, 255, 255, 255});
        DrawRectangle(0, static_cast<int>(layout_.band_top()), static_cast<int>(layout_.window_width()),
                      static_cast<int>(window_layout::band), Color{24, 24, 30, 255});
        // the status line, smaller if it would run past the right edge
        const unsigned int room = layout_.window_width() > 16 ? layout_.window_width() - 16 : 0;
        const unsigned int size = window_layout::text_size(room, static_cast<unsigned int>(
            MeasureText(status_.c_str(), static_cast<int>(window_layout::text))));
        DrawText(status_.c_str(), 8, static_cast<int>(layout_.text_top(size)), static_cast<int>(size),
                 Color{220, 220, 220, 255});
        if (paused_) {
            DrawText("paused", static_cast<int>(layout_.window_width()) - 80, static_cast<int>(layout_.text_top()),
                     static_cast<int>(window_layout::text), Color{255, 220, 40, 255});
        }
        EndDrawing();
    }

    // the screen pixels to a canvas pixel asked for (0: chosen from the size)
    unsigned int pixel_;
    // the geometry of the window, set when it opens
    window_layout layout_{1, 1, window_layout::smallest};
    // the texture the frames go to, and its pixels
    Texture2D texture_{};
    std::vector<unsigned char> pixels_;
    // the status line shown
    std::string status_;
    // whether the window is open, and is paused
    bool open_   = false;
    bool paused_ = false;
    // screenshots saved with S so far
    unsigned int shots_ = 0;
};
} // namespace games
#endif
