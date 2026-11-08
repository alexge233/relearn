# Changes

What each change to relearn did, newest first. The pull requests hold the details, the measurements and the review
of each one; this is the short version.

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
