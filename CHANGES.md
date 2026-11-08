# Changes

What each change to relearn did, newest first. The pull requests hold the details, the measurements and the review
of each one; this is the short version.

## Lessons: demos that explain themselves

- `examples/window/lesson.hpp`, with no graphics in it: the numbers of an update, the formula's columns, the plain
  words, the rules of Q-learning, SARSA and Expected SARSA, `tabular_session` (playing and learning one step at a
  time, behind the `session` interface), the controls, the top bar's layout and clicks.
- `examples/window/studio.hpp`: draws a lesson with raylib, at native resolution: the crawling intro, the clickable
  top bar, the board, the beliefs, the learning curve, the formula with its real numbers, the plain words and the help.
- Sliders for alpha, gamma and epsilon, changing how it learns whilst it runs; the learning curve moves from the
  first step.
- Help bubbles: rest the mouse on anything on the main screen and it says what it is.
- `examples/window/chiptune.hpp`: an original chiptune loop and the sound effects, written as code; M and the top bar
  turn them on and off, `--mute` starts silent; `tests/chiptune_test.cpp` measures the samples.
- A fall is a short scene at low speeds: the walker turns red, a skull bounces up and its ghost floats away.
- The beliefs in a cell are a heatmap, one triangle per move, and it says whether the move was a best guess or a
  random try, which is where epsilon acts.
- `examples/lesson_cliff_walking.cpp`: the first lesson, with Kenney's tiles and a smooth map of the values.
- `examples/window/assets/`: Kenney's Tiny Town and Tiny Dungeon and the Pixel Operator font, all CC0, with credits.
- `tests/lesson_test.cpp`: the numbers on the screen against the library's at every step, a session against
  `games::play_online`, and the words, columns, controls and clicks worked out by hand.
- `docs/lessons/cliff_walking.md`, and the README, the roadmap and `docs/watch.md` point to it.

## Watching the agents play, in a window

- `watched<game_type, screen_type>`: the frames go to a *screen*, `terminal_screen` (the default, what the terminal
  did before, moved out of `watched`) or any class with `fits`, `too_tall` and `show`; closing a screen stops the
  watching.
- `examples/window/layout.hpp`: `window_layout`, the geometry of a window (the pixel size, the band, a minimum width
  with the picture centred, the canvas as bytes), with no raylib in it; `examples/window/window.hpp`:
  `window_screen`, a raylib window opened at the first frame, with pause (space), screenshots (S), a `snapshot`
  of the last frame, and held on its last frame until dismissed unless a snapshot was asked for; `examples/window/CMakeLists.txt`, the sub-project (`-DBUILD_WINDOW=On`) building the seven
  watched examples again as `win_<game>`.
- `--snapshot file.png` on the examples, applied to the screens that take one.
- Tests for the terminal screen, a recording screen through `watched`, the layout and the options; the CI builds the
  window examples on Linux and macOS.
- `docs/watch.md` has the window and two pictures of it, `images/mountain_car_window.png` and
  `images/racetrack_window.png`; the README and the roadmap mark it done.

## Watching the agents play, in the terminal

- `examples/games/watch.hpp`: a `canvas` of pixels rendered two to a character with half blocks, in 24-bit colour
  (`truecolour`) or the 256-colour palette (`colour_256`), chosen from `COLORTERM`; `plot_curve` for a learning
  curve; `view<game_type>`, the drawing hook each watched game specialises, and `has_view` to detect it; and
  `watched<game_type>`, a game that wraps a game and draws it after every step, with a status line, and plays exactly
  as the game it wraps when it is off.
- `examples/games/views.hpp`: the views of Cliff Walking, Windy Gridworld, the Dyna maze, FrozenLake, the racetrack,
  Mountain Car (with the (position, velocity) plane of values, Figure 10.1, when cut into a grid) and cart-pole, the
  cells coloured by their best value through `heat`, and a generic view for any `discretised` game.
- `--watch`, `--delay` and `--every` on seven examples (`cliff_walking`, `windy_gridworld`, `dyna_maze`,
  `frozen_lake`, `racetrack`, `mountain_car`, `cart_pole`), which print exactly what they did without them.
- `games::racetrack` has `rows()` and `columns()`; `games::discretised` has a const `inner()`, `grid()` and
  `place()`.
- `tests/watch_test.cpp`: the palettes, the canvas and what it renders, the curve, the heat, every view pixel by
  pixel, and `watched` playing the very same episodes as the game it wraps, worked out by hand; 71 planted bugs, all
  caught.
- `docs/watch.md`, the README, the roadmap, and a headless `--watch` run in the CI.

## Before this file

Everything up to the planning methods of Chapter 8 (pull requests 1 to 43): the tabular methods of Sutton and
Barto's book, from bandits and dynamic programming to policy gradients, R-Learning and Monte Carlo Tree Search, each
with its game, its tests and its page in `docs/`. The README lists them, and `docs/ROADMAP.md` says what comes next.
