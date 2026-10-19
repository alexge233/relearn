![RELEARN](https://github.com/alexge233/relearn/blob/master/images/relearn.png?raw=true)

[![check](https://github.com/alexge233/relearn/actions/workflows/check.yaml/badge.svg)](https://github.com/alexge233/relearn/actions/workflows/check.yaml)

**English** | [Español](README.es.md)

A Reinforcement Learning *header-only* template library for C++14.
Minimal and simple to use, for a variety of scenarios.
It is based on Sutton and Barto's book [1] and implements *some* of the functionality described in it.

Currently the library implements the following algorithms, each one explained in its own document
(with the formulas, the pseudocode, and how to use it):

- **Q-Learning** (`q_learning`) for deterministic systems, as well as non-deterministic systems (`q_probabilistic`),
  see [docs/q_learning.md](docs/q_learning.md)
- **SARSA** (`sarsa`) and **Expected SARSA** (`expected_sarsa`), which learn the value of the policy the agent
  actually follows, exploration included, see [docs/sarsa.md](docs/sarsa.md)
- **Double Q-Learning** (`double_q_learning`), which keeps two estimates, so that the one picking the best next action
  is not the one valuing it, and isn't fooled by noisy values, see [docs/double_q_learning.md](docs/double_q_learning.md)
- **Dyna-Q** and **Dyna-Q+** (`dyna_q`), which remember every move made, and replay them in their head (planning)
  to learn far more from the same experience, whilst Dyna-Q+ also goes back to check moves it hasn't made for a while,
  in case the world has changed, see [docs/dyna.md](docs/dyna.md)
- **schedules** for the learning rate and the exploration rate, so that each one can *decay* as the agent learns
  (`polynomial_decay`, `exponential_decay` and `linear_decay`), see [docs/schedules.md](docs/schedules.md).
  Do bear in mind that `q_learning`, `sarsa`, `expected_sarsa` and `double_q_learning` can each decay the learning rate
  of every state/action pair on its own

It also comes with a collection of classic games to learn, described in [Games](#games) below.

## Implementation in C++

The template class `policy` stores the learnt values of your states and actions.
A pair of template wrapper classes, `state` and `action`, is provided in the header, which wrap around your own
state and action structures (the `state_trait` and `action_trait`). Those can be anything, provided that:
- they have an `operator==`
- they are hashable, with a specialisation of `std::hash`

Take care to make them unique, two different states whose traits compare equal are treated as one.
The `examples/gridworld_header.hpp` shows how to write them.

You can also create your own `state` and `action` classes and use them instead of the wrappers,
but they must satisfy the below criteria:
- they have an `operator==`
- they have a specialisation of `relearn::hasher`
- states must expose their reward with a method `reward()`

### Markov Decision Process

An MDP is a sequence of States  s<sub>t</sub> linked together by actions a<sub>t</sub>.
At the core of `relearn` is the `policy` which uses MDPs and **maps** them using the rewards `R` received along them.
Whilst multiple episodes may exist, and an episode may be iterated many times, updated, evaluated
and re-experienced, the actual process is done in class `policy`. An overview of an MDP is shown below.

![Markov Decision Process](https://github.com/alexge233/relearn/blob/master/images/mdp.png?raw=true)

The template parameters `state_trait` and `action_trait` are used to describe what a state and action are,
the wrappers are used only for indexing, hashing and policy calculation.

The basic use is to create episodes, denoted as `markov_chain` which you populate as you see fit,
giving each step the reward it earned (any value, positive, negative or zero).
Do bear in mind that each reward belongs to the step that earned it, have a look at the
[reward convention](docs/q_learning.md#reward-convention).

The use of a `markov_chain` defaults to an `std::deque<relearn::link<state_class,action_class>>`
where `relearn::link` is the struct encapsulating a state/action observation.

You may replace the container or link class if you want, but the learners require that:

- the episode has an `operator[]` taking the index of a link, and a `size()` method
- each link has the members `state` and `action`

Basically `std::deque` and `std::vector` work right out of the box (`std::list` won't, as it has no `operator[]`).
The library uses by default `double` for storing and updating Q values,
but you can play around with it if needed.

The class that mostly interests us is `policy` which uses your states and actions,
and stores them in a map of states, where each state (key) has a value of maps (action to value):

```
.
├── s_t
|   ├── a_t: 0.03
|   └── a_t: -0.05
├── s_t
|   ├── a_t: -0.001
|   └── a_t: 0.9
```

You own and control the policy objects, and you can even use multiple ones, however bear in mind
that they are not locked for MT access (you have to do this manually).
Using any of the learners you can then set/update the policy values by iterating your episodes.

# Dependencies

There are **no external dependencies**!
However, your compiler **must support C++14**, so you will need:

- GCC 5 or higher
- Clang 3.4 or higher

These are the first releases with complete C++14 support. The CI builds with GCC on Ubuntu, and with Apple Clang on macOS.

__note__: I haven't tested with a Windows platform!

__note__: building the examples and tests needs CMake 3.5 or higher, and the tests use
[Catch2](https://github.com/catchorg/Catch2) v2, whose single header is downloaded by CMake.

__note__: serialization is optional, and needs Boost (see [enable serialization](#enable-serialization) below).

# Building

There is nothing to build! This is a header-only library, you only need the `src` folder, and to include the **relearn.hpp** header.
**However**, you may compile the examples in order to see how the library is used.

`relearn.hpp` simply includes the parts of the library, which live in `src/relearn/`, one header for each:

| header | what's in it |
|---|---|
| `relearn/core.hpp` | `state`, `action`, the `link` of an episode, and the `policy` |
| `relearn/schedules.hpp` | `polynomial_decay`, `exponential_decay`, `linear_decay`, and the `visit_counter` |
| `relearn/q_learning.hpp` | `q_learning` and `q_probabilistic` |
| `relearn/sarsa.hpp` | `sarsa` and `expected_sarsa` |
| `relearn/double_q_learning.hpp` | `double_q_learning` |
| `relearn/dyna.hpp` | `dyna_q` |

Each one includes whatever it needs, so if you only use one learner (e.g., on a small board, where every byte
and every second of compiling counts) feel free to include just that one, `#include "relearn/q_learning.hpp"`.

To do that, simply:

```bash
mkdir build
cd build
cmake ..
make
```

This builds all the examples, as `ex_<name>` in the `build` folder.

### build tests

```bash
cmake .. -DBUILD_TESTS=On
make
ctest
```

### enable serialization

In order to enable serialization, you need boost and boost-serialization.
See your distro on how to install those.
To enable it, pass a cmake flag:

```bash
cmake .. -DUSING_BOOST_SERIALIZATION=On
```

For example if you want to run the tests **with** serialization:

```bash
cmake .. -DUSING_BOOST_SERIALIZATION=On -DBUILD_TESTS=On
```

You can also use serialization in your **own** project, if you wish to save and load
policies, states or actions: simply define `USING_BOOST_SERIALIZATION` (e.g., `-DUSING_BOOST_SERIALIZATION`)
before including `relearn.hpp`, and use CMake's `find_package` to find, include and link against `boost_serialization`.
Do bear in mind that the `state_trait` (e.g., your state **descriptor**) and the
`action_trait` (e.g., your action **descriptor**) must **also be serializable**.
On how to achieve this, [have a look at this tutorial](http://www.boost.org/doc/libs/1_64_0/libs/serialization/doc/tutorial.html)
if this condition is not met, you will end up with compilation errors.

Because of the flexibility of boost serialization, you can save and load binary, text or xml archives.
Later versions of boost support smart pointers, so even if your descriptors are
`std::shared_ptr` or `std::unique_ptr` you can still save and load them.

# Examples

There is a folder `examples` with two "Gridworld" examples:
- an offline one, which learns from random exploration: `examples/gridworld_offline.cpp` built as `ex_gridworld_offline`
- an online one, which learns after every episode: `examples/gridworld_online.cpp` built as `ex_gridworld_online`

and a collection of classic games, described in [Games](#games) below.

## basic usage

The basic way of using the library is the following:

1. create a class **state**, or use an existing class, structure, or POD which describes (in a *Markovian* sense) your state
2. create a class **action**, or use an existing class, structure, or POD which describes the action.
3. create an *episode* which by default is an `std::deque<relearn::link<state,action>>`, and populate it with the steps taken and the reward each one earned.

At this point, depending on whether you are using an **online** or **offline** algorithm/approach, you have the following options:

4. keep creating episodes, and once you have finished, train the policy with all of them,
**or**
5. every time you create an episode, train your policy with it.

That choice is up to you, and almost always depends on the domain, system or problem you're trying to solve.
It is for this reason that there is no implementation of `on_policy` or `off_policy` or `e_greedy`,
those are very simple algorithms, and are application-specific.
The games show one way of writing them (`games::greedy` and `games::epsilon_greedy` in `examples/games/game.hpp`).

Take a look at the `gridworld` examples, which demonstrate two different ways of achieving the same task.
The `blackjack` example is different: Blackjack is a game of *chance*, where the same decision can win or lose
depending on the cards that follow, so it uses `q_probabilistic`, which learns the expected reward and the
probability of each transition from the games it plays.

## Gridworld

![gridworld image](https://github.com/alexge233/relearn/blob/master/images/gridworld.png?raw=true)

The pinnacle of simplicity when it comes to block/grid world toy problems, our agent resides in a 10x10 gridworld,
which is surrounded by blocks into which it can't move (black colour).
The agent starts at blue (x:1,y:8) and the target is the green (x:1,y:1).
The red blocks are fire/danger/a negative reward, and there is a rudimentary maze.
Moving into the target earns +1, moving into a fire block earns -1, and both end the episode.

Both versions learn with Q-Learning, which learns the values of the best actions, whatever actions
were taken whilst exploring. They differ in how they explore. The offline approach:

- first the agent randomly explores until it can find the positive reward (+1.0) grid block
- then it updates its policies
- finally it follows the best policy learnt

And the online approach:

- the agent explores one episode
- then it updates its policies
- then it tries again, this time following the policies it knows
- it only falls back to random when there does not exist a *positive* best action (or it leads into a wall)
- the entire process is repeated until the goal is discovered.

The actual gridworld is saved in a textfile `examples/gridworld.txt` (feel free to change it).
The header `examples/gridworld_header.hpp` provides the minimal code to demonstrate this staged approach.
The two files:

- `examples/gridworld_offline.cpp`
- `examples/gridworld_online.cpp`

have the different versions of how this task can be solved.

Once we have loaded the world (using function `populate`) we set the start at x:1, y:8 and then
begin the exploration.

### offline q-learning

The offline exploration runs in an infinite loop until the grid block with a **positive** reward is found.
Until that happens, the agent takes a *stochastic* (e.g., random) approach and searches the gridworld.
The function (template parameter `S` is state, and `A` is action):

```cpp
template <typename S, 
          typename A>
std::deque<relearn::link<S,A>> explore(const world & w,
                                       std::mt19937 & gen,
                                       grid start);
```

does the following:

- creates a new episode (e.g., `relearn::markov_chain`)
- starts from the starting gridblock
- randomly picks a direction (see struct `rand_direction` for more)
- records each step with the reward of the block it moved into
- repeats this until either (a) a negative reward has been found (e.g., stepped into a fire block), or (b) the goal block is discovered

Each time the `explore` method is called in the main exploration loop,
an episode is acquired, which is then appended in the list of experienced episodes `std::vector<std::deque<relearn::link<state,action>>> episodes`.

Once the agent has found the target gridblock, it updates its policy memory using those episodes:

```cpp
auto learner = relearn::q_learning<state,action>{0.9, 0.9};
for (int k = 0; k < 200; k++) {
    for (const auto & episode : episodes) {
        learner(episode, policies);
    }
}
```

In this instance, the agent will run two hundred iterations for each episode, since each pass moves the reward
only one step back towards the start, and random episodes can be long. The actual Q-Learning parameters are:
- alpha (learning rate) 0.9
- gamma (discount) 0.9

Finally, once updating policies has finished, the agent proceeds to follow (once) the best action policy:

```cpp
on_policy(w, policies, start);
```

The result is that the agent will go straight from x:1, y:8 to x:1, y:1 avoiding any fire blocks
and unnecessarily searching the gridworld.

This is a __deterministic__ scenario, because the agent knows at any given moment, which action it is taking,
and to __which__ state that action will lead to.

### online q-learning

The online exploration is somewhat different, because the `explore` method does the following:

- creates a new episode
- starts from the starting gridblock
- if a good policy (a Q value higher than zero, for a move that does not lead into a wall) exists, it follows it
- else if no good policy exists, it takes a random action
- repeats until a reward is found, it then **trains** the policies with the latest episode (ten passes)
- an episode that wanders for more than 1000 moves is cut short, and learnt with `terminated = false`
  (see [episodes cut short](docs/q_learning.md#episodes-cut-short)), so that its last move is not learnt as the end of the game

The difference here, is that the actual exploration is instantly affected by what has already been learnt.
In comparison, the offline method is not affected, and may repeat the same sequences over and over again.
However, if the online version stops too early, there is no guarantee that the agent has learnt the
ideal or optimal path to the goal, it could in fact be just a *mediocre* or *silly* path it has discovered.
This of course, is also a problem with offline, where the solution may never be discovered.

Other approaches exist (e.g., ε-Greedy) where the agent may follow the policy,
but randomly choose to ignore it, in order to try and discover a better solution.
The games below use ε-Greedy.

# Games

The folder `examples/games` holds classic reinforcement learning problems, each one in its own header,
and `examples/<game>.cpp` learns each one (built as `ex_<game>`). Most of them come from Sutton and Barto [1],
whereas FrozenLake, cart-pole and mountain car follow Gymnasium's [2] rules and physics, so that agents can later be
compared on both.

Every game has the same shape, after Gymnasium's `reset()`/`step()` (see `examples/games/game.hpp`):

```cpp
games::cliff_walking game;
auto s = game.reset();                        // start an episode
auto out = game.step(games::right);           // out.next, out.reward, out.terminated
auto moves = game.actions(s);                 // the legal actions
```

and `games::play` records an episode in relearn's reward convention, ready for a learner:

```cpp
using types = games::types<games::cliff_walking>;
types::policy memory;
relearn::q_learning<types::state, types::action> learner;
auto r = games::play(game, [&](const auto & s, const auto & legal) {
    return games::epsilon_greedy<games::cliff_walking>(memory, s, legal, 0.1, prng);
}, 1000);                                     // at most 1000 moves
learner(r.episode, memory, r.terminated);     // a cut-short episode is not learnt as finished
```

Alternatively, `games::play_online` plays an episode and learns after every step, as Sutton and Barto's
TD algorithms do. SARSA needs it with a large learning rate (have a look at [docs/sarsa.md](docs/sarsa.md)):

```cpp
auto r = games::play_online(game, agent, learner, memory, 1000);
```

| Game | Source | Learnt with | What it shows |
|---|---|---|---|
| `bandit` | S&B 2.3 | `q_probabilistic` | Exploration vs exploitation, greedy vs ε-Greedy (Figure 2.2) |
| `blackjack` | S&B 5.1 | `q_probabilistic` | A random game, where the learnt strategy matches Figure 5.2 |
| `cliff_walking` | S&B 6.6 | `q_learning`, `sarsa`, `expected_sarsa` | On-policy vs off-policy: Q-Learning learns the path along the cliff edge, SARSA a safer one, and both SARSAs earn more whilst exploring |
| `windy_gridworld` | S&B 6.5, Ex. 6.10 | both Q-Learnings | Steady wind: the 15-move path. Random wind: `q_probabilistic`, and `q_learning` with a learning rate that decays per pair, find the best policy, whereas `q_learning` with α = 0.5 does not |
| `random_walk` | S&B 6.2, 7.1 | `q_probabilistic` | Prediction, with the learnt values next to the exact ones |
| `maximization_bias` | S&B 6.7 | `q_learning`, `double_q_learning` | Q-Learning's overestimation, which Double Q-Learning removes (Figure 6.5) |
| `gamblers_problem` | S&B 4.3 | `q_probabilistic` | Values that match the exact solution |
| `dyna_maze` | S&B 8.1 to 8.3 | `dyna_q` | Planning learns a maze in far fewer episodes (Figure 8.2), and Dyna-Q+ copes with mazes that change (Figures 8.4, 8.5) |
| `tic_tac_toe` | S&B 1.5 | `q_learning` | Against a random or a perfect (minimax) opponent |
| `frozen_lake` | Gymnasium | `q_probabilistic` | Slippery ice, where the learnt 4x4 policy is optimal |
| `cart_pole` | Gymnasium | `q_learning` | A continuous state, cut into boxes (`games::discretised`) |
| `mountain_car` | Gymnasium | `q_learning` | A continuous state, explored by optimistic initial values |

The `tests/games_test.cpp` tests the rules of every game, and that the learners reach the known answer:
either exact solutions (`tests/solvers.hpp`: value iteration, exact blackjack, breadth-first search) or the
optimal results given by the book.

# Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first,
it describes the code style, and how changes are tested.
AI-generated contributions are welcome, as long as they have been reviewed and approved by a human.

# TODO

1. Monte Carlo control: with exploring starts, on-policy, and off-policy with importance sampling (`blackjack`)
2. TD(0) prediction, and n-step TD and n-step SARSA (`random_walk`)
3. eligibility traces: TD(λ), SARSA(λ) and Watkins's Q(λ)
4. REINFORCE, REINFORCE with a baseline, and Actor-Critic, which learn the policy itself (a new short corridor game)
5. the R-Learning continuous algorithm, for tasks that never end (a new access-control queue game)
6. UCB and gradient bandits (`bandit`)
7. Monte Carlo Tree Search (`tic_tac_toe`)
8. prioritized sweeping (`dyna_maze`), and policy iteration and value iteration in the library
9. watch the agents play: a `--watch` mode for the examples, with one drawing hook for two views
   - in the terminal, as good old ASCII art
   - in a window, as retro pixel art: 2D, drawn at a low resolution and scaled up with sharp pixels,
     with sprites and a 16-colour palette defined in the code (so no asset files), a pixel font, optional
     chiptune sounds generated in code, and the learning shown in the same style (a pixel learning
     curve, Q values as tile colours, and a status line)
   - the window would use [raylib](https://www.raylib.com) (zlib licence, Linux and macOS), in an optional
     CMake sub-project which is off by default, so that `relearn.hpp` and the default build keep no dependencies

# References

1. Sutton, R.S. and Barto, A.G., 2018. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 2nd edition. Cambridge, MA: MIT Press.
   Section, example and figure numbers above refer to this edition.
2. [Gymnasium](https://gymnasium.farama.org), the maintained successor of OpenAI Gym.
