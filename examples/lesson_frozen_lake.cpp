/**
 * FrozenLake as a lesson: a traveller crosses slippery ice, where a move goes the way it was meant only one time in three,
 * and learns moves that keep it out of the holes even when it slides.
 *
 * Built by the window sub-project (`cmake .. -DBUILD_WINDOW=On`), as `lesson_frozen_lake`. H shows the controls.
 *
 * @see docs/lessons/frozen_lake.md for the theory
 * @see games/frozen_lake.hpp for the rules
 */
#include "window/frozen.hpp"
#include "window/studio.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lessons = games::lessons;
using lessons::frozen_grid;

/**
 * @class frozen_board
 * @brief draws the lake, the traveller, the splash of a fall, and what the ice can do to a move
 */
class frozen_board
{
public:
    /// @return the height of the board, in screen pixels
    float height() const
    {
        return lake_.rows() * cell();
    }

    /// @return the side of a cell, in screen pixels
    float cell() const
    {
        return static_cast<float>(scale * lessons::studio::tile_side);
    }

    /// @return true: the lake is square, and takes the height of the title
    bool tall() const
    {
        return true;
    }

    /// @return the columns of the board
    int columns() const
    {
        return lake_.columns();
    }

    /// @return the rows of the board
    int rows() const
    {
        return lake_.rows();
    }

    /// @return the names of the moves, in the order of the game's actions
    std::vector<std::string> moves() const
    {
        return {"up", "right", "down", "left"};
    }

    /// @return what the colours and the tiles mean
    const char * legend() const
    {
        return "The traveller starts at the signpost and wants the treasure. The dark cells are holes. The colours run from blue to gold, as it expects to do better. The arrows are its best known move: not the way it wants to go, but the safest thing to try.";
    }

    /// @return what a bubble says about the cell in column @param column and row @param row
    std::string describe(const lessons::session & now,
                         int column,
                         int row) const
    {
        const games::position p{column, row};
        switch (lake_.tile(p)) {
            case 'H': return "A hole. Falling in ends the episode, with nothing.";
            case 'G': return "The treasure. Reaching it is worth 1, and ends the episode.";
            case 'S': return "The start. " + lessons::cell_help(now, p, moves());
            default:  return lessons::cell_help(now, p, moves());
        }
    }

    /// @return what the ice did to the last move, if it got in the way
    std::string aside(const lessons::session & now) const
    {
        return now.latest().valid ? lessons::slip_words(moves(), now.from(), now.last_action(), now.at(), columns(), rows()) : "";
    }

    /// @brief draw the board at (@param x, @param y)
    void draw(const lessons::studio & look,
              const lessons::session & now,
              const lessons::controls & shown,
              float x,
              float y) const
    {
        const float side = cell();
        ice(x, y, side);
        if (shown.values) {
            colours_of_values(now, x, y, side);
        }
        holes(x, y, side);
        if (shown.arrows) {
            arrows_of_policy(look, now, x, y, side);
        }
        look.town(11, 6, x, y, scale);
        look.dungeon(5, 7, x + 3 * side, y + 3 * side, WHITE, scale);
        traveller(look, now, shown, x, y, side);
        DrawRectangleLinesEx({x - 1, y - 1, columns() * side + 2, rows() * side + 2}, 1, lessons::ink::line);
        the_ice_decides(look, now, x + columns() * side, y + 78);
    }

private:
    /// screen pixels to a pixel of the art: only four cells a side, so they can be large
    static constexpr int scale = 5;

    /// @brief the ice, each cell with a little shine
    void ice(float x,
             float y,
             float side) const
    {
        for (int r = 0; r < rows(); r++) {
            for (int c = 0; c < columns(); c++) {
                const int left = static_cast<int>(x + c * side);
                const int top = static_cast<int>(y + r * side);
                const int size = static_cast<int>(side);
                DrawRectangleGradientEx({static_cast<float>(left), static_cast<float>(top), side, side}, Color{196, 228, 248, 255}, Color{150, 200, 236, 255},
                                        Color{178, 216, 244, 255}, Color{226, 244, 255, 255});
                const int shift = (c * 7 + r * 5) % 11;
                DrawLineEx({left + 15.0f + shift, top + 44.0f}, {left + 42.0f + shift, top + 17.0f}, 2, Color{255, 255, 255, 170});
                DrawLineEx({left + 27.0f + shift, top + 63.0f}, {left + 44.0f + shift, top + 46.0f}, 2, Color{255, 255, 255, 130});
                DrawRectangleLines(left, top, size, size, Color{130, 178, 220, 110});
            }
        }
    }

    /// @return true if the value of cell @param p means something
    bool valued(const lessons::session & now,
                const games::position & p) const
    {
        return lake_.tile(p) != 'H' && lake_.tile(p) != 'G' && now.known(p);
    }

    /// @brief the best value of every cell it knows
    void colours_of_values(const lessons::session & now,
                           float x,
                           float y,
                           float side) const
    {
        double low = 0;
        double high = 0;
        bool any = false;
        for (int r = 0; r < rows(); r++) {
            for (int c = 0; c < columns(); c++) {
                if (valued(now, {c, r})) {
                    const double b = now.best({c, r});
                    low = any ? std::min(low, b) : b;
                    high = any ? std::max(high, b) : b;
                    any = true;
                }
            }
        }
        for (int r = 0; any && r < rows(); r++) {
            for (int c = 0; c < columns(); c++) {
                if (valued(now, {c, r})) {
                    DrawRectangle(static_cast<int>(x + c * side), static_cast<int>(y + r * side), static_cast<int>(side), static_cast<int>(side),
                                  lessons::colour(lessons::heat_colour(lessons::scaled(now.best({c, r}), low, high)), 150));
                }
            }
        }
    }

    /// @brief the holes, dark water with a rim of ice
    void holes(float x,
               float y,
               float side) const
    {
        for (int r = 0; r < rows(); r++) {
            for (int c = 0; c < columns(); c++) {
                if (lake_.tile({c, r}) == 'H') {
                    const int cx = static_cast<int>(x + (c + 0.5f) * side);
                    const int cy = static_cast<int>(y + (r + 0.5f) * side);
                    DrawCircleGradient(Vector2{static_cast<float>(cx), static_cast<float>(cy)}, side * 0.46f, Color{10, 26, 60, 255}, Color{34, 80, 130, 255});
                    DrawCircleLines(cx, cy, side * 0.46f, Color{236, 248, 255, 230});
                }
            }
        }
    }

    /// @brief the best known move of every cell where something has been learnt
    void arrows_of_policy(const lessons::studio & look,
                          const lessons::session & now,
                          float x,
                          float y,
                          float side) const
    {
        for (int r = 0; r < rows(); r++) {
            for (int c = 0; c < columns(); c++) {
                if (valued(now, {c, r})) {
                    look.arrow(x + c * side + side / 2, y + r * side + side / 2, now.best_action({c, r}), Color{255, 255, 255, 220}, 15);
                }
            }
        }
    }

    /// @brief the traveller, gliding between cells when it is slow enough to see
    void traveller(const lessons::studio & look,
                   const lessons::session & now,
                   const lessons::controls & shown,
                   float x,
                   float y,
                   float side) const
    {
        if (shown.death.running()) {
            falling(look, now, shown.death, x, y, side);
            return;
        }
        const games::position to = now.at();
        float cx = static_cast<float>(to.x);
        float cy = static_cast<float>(to.y);
        if (shown.gliding()) {
            const auto t = static_cast<float>(shown.progress());
            cx = now.from().x + (to.x - now.from().x) * t;
            cy = now.from().y + (to.y - now.from().y) * t;
        }
        look.dungeon(1, 7, x + cx * side, y + cy * side, WHITE, scale);
    }

    /// @brief the traveller sliding in and turning blue, rings of water, and its ghost floating off
    void falling(const lessons::studio & look,
                 const lessons::session & now,
                 const lessons::death_scene & scene,
                 float x,
                 float y,
                 float side) const
    {
        using part = lessons::death_scene::part;
        const games::position from = now.from();
        const games::position hole = now.at();
        const float left = x + hole.x * side;
        const float top  = y + hole.y * side;
        const auto t = static_cast<float>(scene.along());
        if (scene.shown() == part::walker) {
            const auto fade = static_cast<unsigned char>(255 * (1 - t));
            look.dungeon(1, 7, x + (from.x + (hole.x - from.x) * t) * side, y + (from.y + (hole.y - from.y) * t) * side, Color{fade, fade, 255, 255}, scale);
        }
        else {
            rings(left + side / 2, top + side / 2, side, scene.shown() == part::skull ? t : 1.0f);
        }
        if (scene.shown() == part::ghost) {
            look.dungeon(1, 10, left, top - t * side, Color{255, 255, 255, static_cast<unsigned char>(230 * (1 - t))}, scale);
        }
        look.write("Splash!", left + 13, top + 1, 16, BLACK, true);
        look.write("Splash!", left + 12, top, 16, lessons::ink::danger, true);
    }

    /// @brief rings of water spreading from (@param cx, @param cy), @param t of the way out
    void rings(float cx,
               float cy,
               float side,
               float t) const
    {
        for (int i = 0; i < 3; i++) {
            const float radius = side * 0.12f * static_cast<float>(i + 1) * (0.4f + t);
            DrawCircleLines(static_cast<int>(cx), static_cast<int>(cy), radius, Color{226, 244, 255, static_cast<unsigned char>(240 - i * 60)});
        }
    }

    /// @brief what the ice can do to the move it chose, and what it did, and the colour scale under them
    void the_ice_decides(const lessons::studio & look,
                         const lessons::session & now,
                         float x,
                         float y) const
    {
        const float wide = 576 - columns() * cell();
        const float tall_panel = height() - 78;
        look.panel(x + 16, y, wide - 16, tall_panel, "The ice decides");
        colour_scale(look, x + 32, y + tall_panel - 34, wide - 48);
        if (!now.latest().valid) {
            look.write("Press N for a step.", x + 32, y + 40, 16, lessons::ink::text);
            return;
        }
        const auto names = moves();
        const unsigned int tried = now.last_action();
        const int happened = lessons::turn_of(now.from(), tried, now.at(), columns(), rows());
        look.write("It chose " + names[tried] + ".", x + 32, y + 32, 16, lessons::ink::text, true);
        look.write("One time in three each:", x + 32, y + 50, 16, lessons::ink::text);
        const float box = (wide - 64) / 3;
        int place = 0;
        for (int turn : {-1, 0, 1}) {
            const float left = x + 32 + static_cast<float>(place++) * (box + 8);
            const bool chosen = turn == happened;
            DrawRectangleRounded({left, y + 72, box, 60}, 0.15f, 6, chosen ? Color{84, 72, 40, 255} : Color{44, 46, 80, 255});
            if (chosen) {
                DrawRectangleRoundedLinesEx({left, y + 72, box, 60}, 0.15f, 6, 2, lessons::ink::gold);
            }
            const unsigned int way = lessons::turned(tried, turn);
            look.arrow(left + box / 2, y + 90, way, chosen ? lessons::ink::gold : lessons::ink::text);
            const float word = look.width(names[way], 16, chosen);
            look.write(names[way], left + box / 2 - word / 2, y + 108, 16, chosen ? lessons::ink::gold : lessons::ink::dim, chosen);
        }
        look.write("A value averages all three.", x + 32, y + 140, 16, lessons::ink::dim);
        look.write("A good move is a safe one.", x + 32, y + 158, 16, lessons::ink::dim);
    }

    /// @brief the colours of the values, in a small space
    void colour_scale(const lessons::studio & look,
                      float x,
                      float y,
                      float wide) const
    {
        for (int i = 0; i < static_cast<int>(wide); i++) {
            DrawRectangle(static_cast<int>(x) + i, static_cast<int>(y), 1, 8, lessons::colour(lessons::heat_colour(i / (wide - 1))));
        }
        look.write("worse", x, y + 10, 16, lessons::ink::dim);
        look.write("better", x + wide - look.width("better", 16), y + 10, 16, lessons::ink::dim);
    }

    frozen_grid lake_;
};

int main(int argc,
         char ** argv)
{
    const double alpha = 0.1;
    const double gamma = 0.95;
    const double epsilon = 0.1;
    const std::size_t limit = 100;
    lessons::info about;
    about.title = "FrozenLake";
    about.source = "Gymnasium's FrozenLake-v1";
    about.intro = {"A traveller must cross a frozen lake to reach the treasure on the far side.",
                   "The lake is full of holes, and the ice is slippery. Choose a direction, and only one time in three will you go that way.",
                   "The rest of the time the ice sends you sideways. The same move, from the same place, can win or lose.",
                   "So there is no safe path to memorise. All there is to learn is which move gives the best odds from each cell.",
                   "Watch the arrows. Sometimes the wisest move is to walk away from the treasure."};
    about.watch_for = "The same move can turn out differently, so each value is an average of what the ice has done.";
    std::vector<std::unique_ptr<lessons::session>> learners;
    learners.emplace_back(new lessons::frozen_session<lessons::q_learning_rule<frozen_grid>>(frozen_grid(1), alpha, gamma, epsilon, 1, limit));
    learners.emplace_back(new lessons::frozen_session<lessons::sarsa_rule<frozen_grid>>(frozen_grid(1), alpha, gamma, epsilon, 1, limit));
    learners.emplace_back(new lessons::frozen_session<lessons::expected_sarsa_rule<frozen_grid>>(frozen_grid(1), alpha, gamma, epsilon, 1, limit));
    lessons::lesson_app<frozen_board> app(about, std::move(learners), frozen_board(), lessons::lesson_options::parse(argc, argv));
    return app.run();
}
