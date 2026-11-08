/**
 * Cliff Walking as a lesson: what Q-learning, SARSA and Expected SARSA do, shown one step at a time, each step written
 * out in its formula and in plain words.
 *
 * Built by the window sub-project (`cmake .. -DBUILD_WINDOW=On`), as `lesson_cliff_walking`. H shows the controls.
 *
 * @see docs/lessons/cliff_walking.md for the theory
 * @see games/cliff_walking.hpp for the rules
 */
#include "games/cliff_walking.hpp"
#include "window/studio.hpp"
#include <memory>
#include <string>
#include <vector>

using rules = games::cliff_walking;
namespace lessons = games::lessons;

/**
 * @class cliff_board
 * @brief draws the game: grass, the spikes of the cliff, the signpost, the treasure, the walker, and what it has learnt
 */
class cliff_board
{
public:
    /// @return the height of the board, in screen pixels
    float height() const
    {
        return 4.0f * lessons::studio::cell;
    }

    /// @return the side of a cell, in screen pixels
    float cell() const
    {
        return lessons::studio::cell;
    }

    /// @return false: this board is wide, and fits under the title
    bool tall() const
    {
        return false;
    }

    /// @return nothing: there is nothing to add about a step here
    std::string aside(const lessons::session &) const
    {
        return "";
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
        if (rules::cliff(p)) {
            return "The cliff. Stepping here costs 100 points and sends the walker back to the start.";
        }
        if (p == rules::goal()) {
            return "The treasure. Reaching it ends the episode.";
        }
        const std::string about = lessons::cell_help(now, p, moves());
        return p == rules::start() ? "The start. " + about : about;
    }

    /// @return the names of the moves, in the order of the game's actions
    std::vector<std::string> moves() const
    {
        return {"up", "right", "down", "left"};
    }

    /// @return what the colours and the tiles mean
    const char * legend() const
    {
        return "The walker starts at the signpost and wants the treasure. The spikes are the cliff. The colours run from blue, where it expects to do badly, to gold, where it expects to do well. The arrows are its best known move in each cell.";
    }

    /// @brief draw the board at (@param x, @param y)
    void draw(const lessons::studio & look,
              const lessons::session & now,
              const lessons::controls & shown,
              float x,
              float y) const
    {
        const float side = static_cast<float>(lessons::studio::cell);
        grass(look, x, y);
        if (shown.values) {
            colours_of_values(now, x, y, side);
        }
        cliff(look, x, y, side);
        if (shown.arrows) {
            arrows_of_policy(look, now, x, y, side);
        }
        look.town(11, 6, x, y + 3 * side);
        look.dungeon(5, 7, x + 11 * side, y + 3 * side);
        walker(look, now, shown, x, y, side);
        DrawRectangleLinesEx({x - 1, y - 1, 12 * side + 2, 4 * side + 2}, 1, lessons::ink::line);
    }

private:
    /// @brief the grass, with a few flowers
    void grass(const lessons::studio & look,
               float x,
               float y) const
    {
        const float side = static_cast<float>(lessons::studio::cell);
        for (int r = 0; r < rules::height; r++) {
            for (int c = 0; c < rules::width; c++) {
                const int kind = (c * 7 + r * 3) % 5 == 0 ? 1 : ((c * 5 + r) % 11 == 0 ? 2 : 0);
                look.town(kind, 0, x + c * side, y + r * side);
            }
        }
    }

    /// @return the best value of the cell (@param c, @param r), kept on the board and away from the cliff, from 0 to 1
    double smooth(const lessons::session & now,
                  int c,
                  int r,
                  double low,
                  double high) const
    {
        games::position p{std::max(0, std::min(rules::width - 1, c)), std::max(0, std::min(rules::height - 1, r))};
        if (rules::cliff(p)) {
            p.y--;
        }
        return lessons::scaled(now.known(p) ? now.best(p) : low, low, high);
    }

    /// @brief the values as a smooth surface: each corner of a cell is the mean of the four cells around it
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
                if (rules::cliff(p) || p == rules::goal() || !now.known(p)) {
                    continue;
                }
                const double b = now.best(p);
                low = any ? std::min(low, b) : b;
                high = any ? std::max(high, b) : b;
                any = true;
            }
        }
        if (!any) {
            return;
        }
        const auto corner = [&](int c, int r) {
            const double mean = (smooth(now, c - 1, r - 1, low, high) + smooth(now, c, r - 1, low, high) +
                                 smooth(now, c - 1, r, low, high) + smooth(now, c, r, low, high)) / 4;
            return lessons::colour(lessons::heat_colour(mean), 165);
        };
        for (int r = 0; r < rules::height; r++) {
            for (int c = 0; c < rules::width; c++) {
                if (!rules::cliff({c, r})) {
                    DrawRectangleGradientEx({x + c * side, y + r * side, side, side}, corner(c, r), corner(c, r + 1), corner(c + 1, r + 1), corner(c + 1, r));
                }
            }
        }
    }

    /// @brief the cliff: a dark drop, with spikes
    void cliff(const lessons::studio & look,
               float x,
               float y,
               float side) const
    {
        for (int c = 1; c < rules::width - 1; c++) {
            DrawRectangleGradientV(static_cast<int>(x + c * side), static_cast<int>(y + 3 * side), static_cast<int>(side), static_cast<int>(side),
                                   Color{70, 30, 40, 255}, Color{14, 8, 20, 255});
            look.dungeon(5, 3, x + c * side, y + 3 * side);
        }
    }

    /// @brief the best known move of every cell it has been to
    void arrows_of_policy(const lessons::studio & look,
                          const lessons::session & now,
                          float x,
                          float y,
                          float side) const
    {
        for (int r = 0; r < rules::height; r++) {
            for (int c = 0; c < rules::width; c++) {
                const games::position p{c, r};
                if (!rules::cliff(p) && p != rules::goal() && now.known(p)) {
                    look.arrow(x + c * side + side / 2, y + r * side + side / 2, now.best_action(p), Color{255, 255, 255, 190});
                }
            }
        }
    }

    /// @brief the walker, gliding between cells when it is slow enough to see, and what a fall cost it
    void walker(const lessons::studio & look,
                const lessons::session & now,
                const lessons::controls & shown,
                float x,
                float y,
                float side) const
    {
        if (shown.death.running()) {
            dying(look, now, shown.death, x, y, side);
            caption(look, now, x, y, side);
            return;
        }
        const games::position to = now.at();
        float cx = static_cast<float>(to.x);
        float cy = static_cast<float>(to.y);
        if (shown.gliding() && !now.punished()) {
            const auto t = static_cast<float>(shown.progress());
            cx = now.from().x + (to.x - now.from().x) * t;
            cy = now.from().y + (to.y - now.from().y) * t;
        }
        look.dungeon(0, 8, x + cx * side, y + cy * side);
        caption(look, now, x, y, side);
    }

    /// @brief what a fall cost it
    void caption(const lessons::studio & look,
                 const lessons::session & now,
                 float x,
                 float y,
                 float side) const
    {
        if (now.punished()) {
            look.write("-100! Back to the start.", x + 57, y + 3 * side - 23, 16, BLACK, true);
            look.write("-100! Back to the start.", x + 56, y + 3 * side - 24, 16, lessons::ink::danger, true);
        }
    }

    /// @brief the walker going onto the spikes and turning red, a skull bouncing up, and its ghost floating away
    void dying(const lessons::studio & look,
               const lessons::session & now,
               const lessons::death_scene & scene,
               float x,
               float y,
               float side) const
    {
        using part = lessons::death_scene::part;
        const games::position from = now.from();
        const games::position spikes = lessons::moved_to(from, now.last_action());
        const float left = x + spikes.x * side;
        const float top  = y + spikes.y * side;
        const auto t = static_cast<float>(scene.along());
        if (scene.shown() == part::walker) {
            const auto fade = static_cast<unsigned char>(255 * (1 - t));
            look.dungeon(0, 8, x + (from.x + (spikes.x - from.x) * t) * side, y + (from.y + (spikes.y - from.y) * t) * side, Color{255, fade, fade, 255});
        }
        else if (scene.shown() == part::skull) {
            look.skull(left, top - std::sin(t * PI) * side * 0.3f);
        }
        else {
            look.skull(left, top);
            look.dungeon(1, 10, left, top - t * side, Color{255, 255, 255, static_cast<unsigned char>(230 * (1 - t))});
        }
    }
};

int main(int argc,
         char ** argv)
{
    const double alpha = 0.5;
    const double gamma = 1.0;
    const double epsilon = 0.1;
    lessons::info about;
    about.title = "Cliff Walking";
    about.source = "Sutton and Barto, Example 6.6";
    about.intro = {"A lone walker must cross from the signpost to the treasure.",
                   "Every step costs one point. One step off the cliff costs a hundred, and it is back to the start.",
                   "Nobody tells the walker the way. All it has is a table of guesses: how much it believes each move is worth.",
                   "After every step it compares what it believed with what really happened, and corrects the guess a little.",
                   "That one small formula, repeated, is all there is. Watch the numbers draw the path."};
    about.watch_for = "Repeat this at every step, and good values spread back from the treasure: watch the colours.";
    std::vector<std::unique_ptr<lessons::session>> learners;
    learners.emplace_back(new lessons::tabular_session<rules, lessons::q_learning_rule<rules>>(rules(), alpha, gamma, epsilon, 1));
    learners.emplace_back(new lessons::tabular_session<rules, lessons::sarsa_rule<rules>>(rules(), alpha, gamma, epsilon, 1));
    learners.emplace_back(new lessons::tabular_session<rules, lessons::expected_sarsa_rule<rules>>(rules(), alpha, gamma, epsilon, 1));
    lessons::lesson_app<cliff_board> app(about, std::move(learners), cliff_board(), lessons::lesson_options::parse(argc, argv));
    return app.run();
}
