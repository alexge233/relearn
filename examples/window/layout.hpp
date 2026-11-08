#ifndef RELEARN_WINDOW_LAYOUT_HPP
#define RELEARN_WINDOW_LAYOUT_HPP
/**
 * How a frame is laid out in a window: the canvas scaled up, every pixel a square of `pixel` x `pixel` screen pixels,
 * sharp, and the status line in a band under it. Nothing here needs raylib, so that it can be tested anywhere;
 * window.hpp draws the layout with it.
 */
#include "../games/watch.hpp"
#include <algorithm>
#include <vector>

namespace games
{
/**
 * @class window_layout
 * @brief the geometry of a window showing a canvas of `width` x `height` pixels
 */
class window_layout
{
public:
    /// the height of the band under the picture, holding the status line, in screen pixels
    static constexpr unsigned int band = 28;
    /// the size of the text of the status line, in screen pixels
    static constexpr unsigned int text = 18;
    /// the window is kept within this many screen pixels, unless the canvas is bigger even at `smallest`
    static constexpr unsigned int widest  = 960;
    static constexpr unsigned int tallest = 640;
    /// the fewest screen pixels to a canvas pixel
    static constexpr unsigned int smallest = 2;
    /// the most
    static constexpr unsigned int largest = 16;
    /// the window is at least this wide, so that the status line has room; a narrower picture is centred
    static constexpr unsigned int narrowest = 480;

    /**
     * @brief the layout of a canvas of @param width x @param height pixels
     * @param pixel is the screen pixels to a canvas pixel: 0 to choose the largest that keeps the window within
     *        `widest` x `tallest`, between `smallest` and `largest`
     */
    window_layout(unsigned int width,
                  unsigned int height,
                  unsigned int pixel = 0)
    : width_(width),
      height_(height),
      pixel_(pixel > 0 ? pixel : pixel_size(width, height))
    {}

    /// @return the largest number of screen pixels to a canvas pixel that keeps a canvas of @param width x
    ///         @param height within `widest` x `tallest` (the band included), `smallest` at least
    static unsigned int pixel_size(unsigned int width,
                                   unsigned int height)
    {
        // local copies: std::min/max take references, which C++14 static constexpr members cannot bind to
        // without a separate definition
        const unsigned int fewest = smallest;
        const unsigned int most   = largest;
        const unsigned int by_width  = width > 0 ? widest / width : most;
        const unsigned int by_height = height > 0 ? (tallest - band) / height : most;
        return std::max(fewest, std::min(most, std::min(by_width, by_height)));
    }

    /// @return the screen pixels to a canvas pixel
    unsigned int pixel() const
    {
        return pixel_;
    }

    /// @return the width of the window, in screen pixels: the picture's, or `narrowest` if that is more
    unsigned int window_width() const
    {
        const unsigned int least = narrowest;
        return std::max(width_ * pixel_, least);
    }

    /// @return the left edge of the picture: 0, or half the room left in a window wider than it
    unsigned int picture_left() const
    {
        return (window_width() - width_ * pixel_) / 2;
    }

    /// @return the height of the window, in screen pixels: the picture and the band
    unsigned int window_height() const
    {
        return height_ * pixel_ + band;
    }

    /// @return the top of the band, where the status line is written
    unsigned int band_top() const
    {
        return height_ * pixel_;
    }

    /// @return the top of the status line's text, of size @param size: in the band, centred
    unsigned int text_top(unsigned int size = text) const
    {
        return band_top() + (band - std::min(size, band)) / 2;
    }

    /// the smallest the status line's text goes, in screen pixels, when it is too long for the window
    static constexpr unsigned int smallest_text = 10;

    /// @return the size of the status line's text, when at size `text` it measures @param measured screen pixels
    ///         and the band has @param available: `text` if it fits, else smaller in proportion, `smallest_text` at least
    static unsigned int text_size(unsigned int available,
                                  unsigned int measured)
    {
        const unsigned int full  = text;
        const unsigned int least = smallest_text;
        if (measured <= available) {
            return full;
        }
        return std::max(least, full * available / std::max(measured, 1u));
    }

    /// @return the pixels of @param c as bytes, red, green and blue of each one, row by row: what a texture takes
    static std::vector<unsigned char> bytes(const canvas & c)
    {
        std::vector<unsigned char> out;
        out.reserve(static_cast<std::size_t>(c.width()) * c.height() * 3);
        for (int y = 0; y < static_cast<int>(c.height()); y++) {
            for (int x = 0; x < static_cast<int>(c.width()); x++) {
                const rgb p = c.at(x, y);
                out.push_back(p.r);
                out.push_back(p.g);
                out.push_back(p.b);
            }
        }
        return out;
    }

private:
    // the size of the canvas, in its pixels
    unsigned int width_;
    unsigned int height_;
    // the screen pixels to a canvas pixel
    unsigned int pixel_;
};
} // namespace games
#endif
