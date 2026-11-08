# The Dyna Maze: a lesson

A wizard learns its way through a maze, and between its real steps it imagines others, from what it remembers. It
learns from those with the very same formula.

```bash
mkdir build && cd build
cmake .. -DBUILD_WINDOW=On
make lesson_dyna_maze
./lesson_dyna_maze
```

## The task

A maze of 9 by 6 cells, from the signpost on the left to the treasure in the top right corner (Sutton and Barto,
Example 8.1). Walls and the border keep the wizard where it is.

- A **state** `s` is the cell the wizard is in.
- An **action** `a` is a move: up, right, down or left.
- The **reward** `r` is 1 for reaching the treasure, which ends the episode, and 0 for everything else. The discount
  `γ = 0.95` is what makes a shorter way better.

The shortest way is 14 moves. Learning with so little reward is slow: after a first walk of about a thousand moves,
one single value changes, the one next to the treasure.

## What Dyna-Q adds

After each real step, it does three things:

1. It learns from the step, as Q-learning does.
2. It **remembers** it: taking `a` in `s` gave `r` and led to `s'`.
3. It **plans**, `n` times: it picks a cell it has been in and a move it made there, at random, asks its memory what
   happened, and learns from that with the same formula.

```
Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') - Q(s,a) ]
```

So one real value can spread all the way back along the path during a single pause, instead of one cell per episode.

## What is on the screen

- **The maze**: the wizard moves only on real steps. **The ghost** is an imagined step, going from the cell it picked to
  where its memory says that led. The colours are how good it believes each cell is, and the arrows its best known moves.
- **What it remembers**: a map of the cells it has been in, the only ones it can imagine, and how many real and
  imagined steps it has made.
- **Its beliefs in that cell**, for the cell of the last step, real or imagined.
- **Learning curve**: the real moves each episode took. It should fall.
- **The formula**, with the real numbers of this step. Its title says when the step was imagined, and which one.
- **In plain words**: the same step, and where its facts came from.
- **Try other values**: α, γ, ε, and `n`, from 0 to 50, from the next real step on.

The controls are those of [Cliff Walking](cliff_walking.md#controls). The top bar's list and the Tab key switch between
`n = 5`, `n = 0` (plain Q-learning) and `n = 50`, each with its own memory.

## What to try

1. Run `n = 5` slowly, and watch the first episode. Nothing is learnt until the treasure, so the ghosts change nothing.
2. Watch the second episode: the ghosts near the treasure start to carry its value backwards.
3. Press Tab to reach `n = 0`, then `n = 50`, and compare the curves. The book's Figure 8.2 is this comparison.
4. Drag `n` to 50 and raise the speed: the colours fill the maze in a few real moves.

## Questions people ask

**Is Dyna-Q a special version of Q-learning?** Yes: Q-learning with a memory. With `n = 0` it is exactly Q-learning. The
maze is only the small world from the book that shows it off.

**So it has an episodic memory?** Not quite. It does not keep episodes or their order, only one entry per cell and move:
"last time I did this here, I got this and ended up there". Each new visit overwrites the old one, and replay picks
entries at random, more like shuffling flashcards than reliving a journey. Values still spread backwards, because
neighbouring entries keep being replayed.

**Then it improves its policy without exploring?** It improves it without new experience, but it can discover nothing new. Replay
works out what the facts it has imply: it can never imagine a cell it has not been in, or a move it has not made there. So it
must still explore for real. Exploring gathers facts, and planning works out what they imply.

**What trains what?** The word "model" is the confusing one. In reinforcement learning, the model is the memory, the facts
about how the world behaves. What is trained is the table `Q`, from two sources with one formula:

| | What it is | What it does |
|---|---|---|
| Real step | Moving in the world | Updates `Q`, and writes to the memory |
| Memory (the model) | Facts about what happened | Supplies the imagined steps |
| Imagined step | Replaying a fact | Updates `Q`, and nothing else |
| `Q` | How good each move is believed to be | Decides which move to make |

Exploring is only part of the real steps: 90% of them here exploit, taking the best known move, and 10% explore at random.
Both kinds feed the memory.

**How does this relate to DQN?** DQN is Q-learning with a neural network in place of the table, so that it can cope with
far too many states to list. It keeps a replay buffer of recent steps and trains on random batches of them, which is Dyna's
trick at scale; random batches also stop consecutive, nearly identical frames from upsetting the network. Be careful with the words:
in deep learning, "the model" means the network, but DQN is called *model-free*, as its network only predicts values, not what
the world will do next. Dreamer and MuZero do learn a neural model of the world, and train inside it.

## How it is made, and tested

`relearn::dyna_q` tells `on_imagined` what each imagined step did. `examples/window/planning.hpp` runs the library's
planning one step at a time and shows what it is told, so nothing in the lesson computes an update.
`examples/lesson_dyna_maze.cpp` is this lesson's board and text.

`tests/planning_lesson_test.cpp` checks that a session learns exactly what the library's own loop does for `n` = 0, 5 and
50 from the same seed, that `n = 0` is Q-learning, that the formula shown holds at every step, real or imagined, against
the value in the policy, that what it remembers of a move is what the game does, and that planning needs far fewer real
moves and ends on the 14-move path.

## References

Sutton and Barto, *Reinforcement Learning: An Introduction*, second edition, Section 8.2 and Example 8.1.
