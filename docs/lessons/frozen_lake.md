# FrozenLake: a lesson

A traveller crosses a frozen lake to the treasure. The ice is slippery, so the same move from the same cell can win or lose.

```bash
mkdir build && cd build
cmake .. -DBUILD_WINDOW=On
make lesson_frozen_lake
./lesson_frozen_lake
```

## The task

The lake is 4 by 4 cells, with Gymnasium's rules and map (`FrozenLake-v1`).

- A **state** `s` is the cell the traveller is in.
- An **action** `a` is the move it *chooses*: up, right, down or left.
- The **ice** decides what happens: the move goes as chosen one time in three, and to either side of it one time in three each.
- The **reward** is 1 for reaching the treasure. Falling into a hole ends the episode with nothing, and so does nothing else.

## What is new here

The other lessons are certain: the same move from the same cell always does the same thing. Here it does not, and that
changes what a value means:

- `Q(s, a)` is an **average** over everything the ice can do to that move. The small steps of `α` are what make it one.
- A good move is a **safe** move, not the one pointing at the treasure. The arrows are what to *try*, and sometimes the
  best one points away from where it wants to go, because the ice can then only send it somewhere harmless. In the cell
  under the start, with a hole to its right, the best move is left.
- Even the best policy loses about a quarter of the time. The curve is the share won of the last 100 episodes, because one
  episode says nothing.

## What is on the screen

- **The lake**, with the colours of the values and the arrows of its best known moves.
- **The ice decides**: the move it chose, the three things the ice could do with it, and which it did.
- **In plain words**: when the ice gets in the way, it says so, in orange.
- The rest is as in [Cliff Walking](cliff_walking.md#what-is-on-the-screen). The top bar and Tab switch between Q-learning,
  SARSA and Expected SARSA.

## What to try

1. Run it slowly, and watch "The ice decides". Count how often the move goes as chosen.
2. Raise the speed until the curve climbs, and find the arrows that point away from the treasure.
3. Set α to 1 and press Z. Each value then becomes whatever happened last, with no averaging: it wins about 2% of the time, against about 63% at α = 0.1.
4. Set ε to 0 and press Z. With nothing to explore, it settles on a poor plan early and wins about 4% of the time.

## How it is made, and tested

`examples/window/frozen.hpp` adapts the game to the lessons: `frozen_lake` numbers its moves left, down, right, up, and the
lessons use up, right, down, left. `tests/frozen_lesson_test.cpp` checks that the moves really go a third of the time each way,
that the formula on the screen holds at every step for each learner, that a session learns exactly what `games::play_online`
does from the same seed, and that after 20,000 episodes the best moves it has learnt win more than 55% of the games (73% in that run; about 63% on average after 5,000).

## References

Gymnasium, `FrozenLake-v1`; Sutton and Barto, *Reinforcement Learning: An Introduction*, second edition, Chapter 6.
