/**
 * The Dyna maze as a lesson: a wizard learns its way through a maze, and between its real steps imagines others, from
 * what it remembers, learning from them with the very same formula (Sutton and Barto, Example 8.1).
 *
 *     ./lesson_dyna_maze               # H shows the controls
 *
 * Build it with `-DBUILD_WINDOW=On` (have a look at docs/lessons/dyna_maze.md).
 */
#include "games/dyna_maze.hpp"
#include "window/studio.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lessons = games::lessons;
using rules = games::dyna_maze;

/**
 * @class maze_board
 * @brief draws the game, with the wizard, its ghost when it imagines, and what it remembers beside them
 */
class maze_board
{
public:
    /// @return the height of the board, in screen pixels
    float height() const
    {
        return rules::height * cell();
    }

    /// @return the side of a cell, in screen pixels
    float cell() const
    {
        return static_cast<float>(scale * lessons::studio::tile_side);
    }

    /// @return the columns of the board
    int columns() const
    {
        return rules::width;
    }

    /// @return the rows of the board
    int rows() const
    {
        return rules::height;
    }

    /// @return what a bubble says about the cell in column @param column and row @param row
    std::string describe(const lessons::session & now,
                         int column,
                         int row) const
    {
        const games::position p{column, row};
        if (maze_.wall(p)) {
            return "A wall. Walking into it wastes a move.";
        }
        if (p == maze_.goal()) {
            return "The treasure. Reaching it is worth 1, the only reward there is.";
        }
        const std::string about = lessons::cell_help(now, p, moves());
        return p == maze_.start() ? "The start. " + about : about;
    }

    /// @return the names of the moves, in the order of the game's actions
    std::vector<std::string> moves() const
    {
        return {"up", "right", "down", "left"};
    }

    /// @return what the colours and the tiles mean
    const char * legend() const
    {
        return "The wizard starts at the signpost and wants the treasure. The colours run from blue to gold, as it expects to do better. Its ghost is a step it is only imagining. Beside the maze is what it remembers.";
    }

    /// @brief draw the board at (@param x, @param y)
    void draw(const lessons::studio & look,
              const lessons::session & now,
              const lessons::controls & shown,
              float x,
              float y) const
    {
        const float side = cell();
        floor(look, x, y, side);
        if (shown.values) {
            colours_of_values(now, x, y, side);
        }
        if (shown.arrows) {
            arrows_of_policy(look, now, x, y, side);
        }
        look.town(11, 6, x + maze_.start().x * side, y + maze_.start().y * side, scale);
        look.dungeon(5, 7, x + maze_.goal().x * side, y + maze_.goal().y * side, WHITE, scale);
        wizard(look, now, shown, x, y, side);
        DrawRectangleLinesEx({x - 1, y - 1, columns() * side + 2, rows() * side + 2}, 1, lessons::ink::line);
        memory(look, now, x + columns() * side + 16, y);
    }

private:
    /// screen pixels to a pixel of the art: the maze is taller than the cliff, so its tiles are smaller
    static constexpr int scale = 2;
    /// the side of a cell of the map of what it remembers
    static constexpr int map_cell = 14;

    /// @brief the floor and the walls
    void floor(const lessons::studio & look,
               float x,
               float y,
               float side) const
    {
        for (int r = 0; r < rules::height; r++) {
            for (int c = 0; c < rules::width; c++) {
                const bool wall = maze_.wall({c, r});
                look.dungeon(wall ? 9 : 2, 4, x + c * side, y + r * side, WHITE, scale);
            }
        }
    }

    /// @brief the best value of every cell it knows, as flat colours: there is no smoothing across a wall
    void colours_of_values(const lessons::session & now,
                           float x,
                           float y,
                           float side) const
    {
        double low = 0;
        double high = 0;
        bool any = false;
        for (int r = 0; r < rules::height; r++) {
            for (int c = 0; c < rules::width; c++) {
                const games::position p{c, r};
                if (!maze_.wall(p) && p != maze_.goal() && now.known(p)) {
                    const double b = now.best(p);
                    low = any ? std::min(low, b) : b;
                    high = any ? std::max(high, b) : b;
                    any = true;
                }
            }
        }
        for (int r = 0; any && r < rules::height; r++) {
            for (int c = 0; c < rules::width; c++) {
                const games::position p{c, r};
                if (!maze_.wall(p) && p != maze_.goal() && now.known(p)) {
                    DrawRectangle(static_cast<int>(x + c * side), static_cast<int>(y + r * side), static_cast<int>(side), static_cast<int>(side),
                                  lessons::colour(lessons::heat_colour(lessons::scaled(now.best(p), low, high)), 170));
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
        for (int r = 0; r < rules::height; r++) {
            for (int c = 0; c < rules::width; c++) {
                const games::position p{c, r};
                if (!maze_.wall(p) && p != maze_.goal() && now.known(p)) {
                    look.arrow(x + c * side + side / 2, y + r * side + side / 2, now.best_action(p), Color{255, 255, 255, 190});
                }
            }
        }
    }

    /// @brief the wizard, which only moves on real steps, and its ghost, which is an imagined step
    void wizard(const lessons::studio & look,
                const lessons::session & now,
                const lessons::controls & shown,
                float x,
                float y,
                float side) const
    {
        const lessons::planner & plans = *now.plans();
        const games::position to = now.at();
        float cx = static_cast<float>(to.x);
        float cy = static_cast<float>(to.y);
        if (plans.imagining()) {
            ghost(look, now, plans, shown, x, y, side);
        }
        else if (shown.gliding()) {
            const auto t = static_cast<float>(shown.progress());
            cx = now.from().x + (to.x - now.from().x) * t;
            cy = now.from().y + (to.y - now.from().y) * t;
        }
        look.dungeon(0, 7, x + cx * side, y + cy * side, WHITE, scale);
    }

    /// @brief the ghost of the wizard, going from the cell it picked to where its memory says that led
    void ghost(const lessons::studio & look,
               const lessons::session & now,
               const lessons::planner & plans,
               const lessons::controls & shown,
               float x,
               float y,
               float side) const
    {
        const games::position from = now.from();
        const games::position to = plans.imagined_to();
        const Vector2 start{x + (from.x + 0.5f) * side, y + (from.y + 0.5f) * side};
        const Vector2 end{x + (to.x + 0.5f) * side, y + (to.y + 0.5f) * side};
        DrawLineEx(start, end, 2, Color{170, 210, 255, 200});
        const float t = shown.gliding() ? static_cast<float>(shown.progress()) : 1.0f;
        look.dungeon(0, 7, x + (from.x + (to.x - from.x) * t) * side, y + (from.y + (to.y - from.y) * t) * side, Color{150, 200, 255, 170}, scale);
    }

    /// @brief what it remembers: a small map of the cells, and how much it has really done against how much it imagined
    void memory(const lessons::studio & look,
                const lessons::session & now,
                float x,
                float y) const
    {
        const lessons::planner & plans = *now.plans();
        look.panel(x, y, 272, height(), "What it remembers");
        const float left = x + 16;
        const float top = y + 36;
        for (int r = 0; r < rules::height; r++) {
            for (int c = 0; c < rules::width; c++) {
                const games::position p{c, r};
                const Color shade = maze_.wall(p) ? Color{14, 14, 26, 255} : (plans.remembers(p) ? Color{120, 170, 255, 255} : Color{58, 62, 96, 255});
                DrawRectangle(static_cast<int>(left) + c * map_cell, static_cast<int>(top) + r * map_cell, map_cell - 1, map_cell - 1, shade);
            }
        }
        spot(left, top, now.at(), lessons::ink::gold);
        if (plans.imagining()) {
            spot(left, top, now.from(), WHITE);
        }
        const float side_x = left + 9 * map_cell + 12;
        look.write("real", side_x, top, 16, lessons::ink::dim);
        look.write(std::to_string(plans.real_steps()), side_x, top + 18, 16, lessons::ink::gold, true);
        look.write("imagined", side_x, top + 44, 16, lessons::ink::dim);
        look.write(std::to_string(plans.imagined_steps()), side_x, top + 62, 16, lessons::ink::belief, true);
        look.write("Light cells are ones it has been in.", x + 16, top + 94, 16, lessons::ink::dim);
        look.write("It can only imagine those.", x + 16, top + 112, 16, lessons::ink::dim);
    }

    /// @brief mark the cell @param p on the map at (@param left, @param top)
    void spot(float left,
              float top,
              const games::position & p,
              Color c) const
    {
        DrawRectangle(static_cast<int>(left) + p.x * map_cell + 3, static_cast<int>(top) + p.y * map_cell + 3, map_cell - 7, map_cell - 7, c);
    }

    rules maze_;
};

int main(int argc,
         char ** argv)
{
    const double alpha = 0.1;
    const double gamma = 0.95;
    const double epsilon = 0.1;
    lessons::info about;
    about.title = "The Dyna Maze";
    about.source = "Sutton and Barto, Example 8.1";
    about.intro = {"A wizard is lost in a maze, with a treasure somewhere in the far corner.",
                   "Only the treasure is worth anything. Every other step is worth nothing at all.",
                   "So at first, learning is painfully slow: a single value, at the very end of a very long walk.",
                   "But this wizard has a memory. After every real step, it closes its eyes and replays moves it has already made.",
                   "It learns from those dreams with exactly the same formula. Watch how few real steps it comes to need."};
    about.watch_for = "Each ghost is a step that never happened, yet it teaches as much as a real one.";
    std::vector<std::unique_ptr<lessons::session>> learners;
    learners.emplace_back(new lessons::dyna_session<rules>(rules(), alpha, gamma, epsilon, 5, 1));
    learners.emplace_back(new lessons::dyna_session<rules>(rules(), alpha, gamma, epsilon, 0, 1));
    learners.emplace_back(new lessons::dyna_session<rules>(rules(), alpha, gamma, epsilon, 50, 1));
    lessons::lesson_app<maze_board> app(about, std::move(learners), maze_board(), lessons::lesson_options::parse(argc, argv));
    return app.run();
}
