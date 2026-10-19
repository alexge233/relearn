![RELEARN](https://github.com/alexge233/relearn/blob/master/images/relearn.png?raw=true)

[![check](https://github.com/alexge233/relearn/actions/workflows/check.yaml/badge.svg)](https://github.com/alexge233/relearn/actions/workflows/check.yaml)

A Reinforcement Learning *header-only* template library for C++14.
Minimal and simple to use, for a variety of scenarios.
It is based on Sutton and Barto's book [1] and implements *some* of the functionality described in it.

The library currently implements four learners:

- `q_learning`: Q-learning, which updates from each step it is shown, with a learning rate α.
- `q_probabilistic`: Q-learning for non-deterministic systems, which learns the expected reward and the
  transition probabilities from experience, and needs no learning rate.
- `sarsa`: SARSA, which learns the value of the policy the agent follows, exploration included
  (see [docs/sarsa.md](docs/sarsa.md)).
- `expected_sarsa`: Expected SARSA, which learns like SARSA, but from the value the next state is
  expected to have under the agent's policy, so it learns more steadily (see [docs/sarsa.md](docs/sarsa.md)).

The learning rate of `q_learning`, `sarsa` and `expected_sarsa` can decay as each state/action pair
is updated, and the exploration rate can fade as the agent learns, using the schedules
`polynomial_decay`, `exponential_decay` and `linear_decay` (see [docs/schedules.md](docs/schedules.md)).

It also comes with a collection of classic games to learn, described in [Games](#games) below.

## Q-learning

The update rule of `q_learning` is:

```math
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \left[ r_t + \gamma \max_a Q(s_{t+1},a) - Q(s_t,a_t) \right]
```

and for the last step of an episode, which has no next state:

```math
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \left[ r_t - Q(s_t,a_t) \right]
```

where `r_t` is the reward received after taking action `a_t` in state `s_t` (see [Reward convention](#reward-convention)).
It works in deterministic systems, and also in non-deterministic ones provided every state/action pair keeps
being visited, and the learning rates used on its visits satisfy Σα = ∞ and Σα² < ∞ (for example α = 1/n
after n visits). A rate that decays faster, such as α = 1/n², stops learning before the values are reached.

The policy decision is entirely up to you to decide; in the case of a deterministic system,
usually the options are to either stay on policy, taking the action of highest value:

```math
a_t = \pi(s_t) = \arg\max_a Q(s_t,a)
```

or, using ε-greedy, to take a random action with some (often *decaying*) probability ε,
and the action of highest value otherwise.

## Q-learning for non-deterministic systems

In a non-deterministic system, taking action `a_t` in state `s_t` does not always lead to the same next state,
or earn the same reward. `q_probabilistic` remembers, for every state and action it has seen:

- how many times `n(s_t,a_t)` the action was taken in that state,
- the mean reward received after it, `r̄(s_t,a_t)`,
- how many times `n(s_t,a_t,s_{t+1})` each next state followed.

From these it estimates the probability of each transition, **observed** after being taken:

```math
P(s_{t+1} \mid s_t,a_t) = \frac{n(s_t,a_t,s_{t+1})}{n(s_t,a_t)}
```

and sets the value to the expected reward plus the discounted value of the expected next state:

```math
Q(s_t,a_t) = \bar{r}(s_t,a_t) + \gamma \sum_{s_{t+1}} P(s_{t+1} \mid s_t,a_t) \max_a Q(s_{t+1},a)
```

An episode that ends after `(s_t, a_t)` has no next state and adds nothing to the sum.

### Reward convention

In an episode, each link `(s_t, a_t)` carries in `s_t.reward()` the reward received *after* taking `a_t` in `s_t`.
This is the `r_t` above, which Sutton & Barto (2nd ed.) write as `R_{t+1}`: the same reward, a different index.
The last link of an episode is the final decision, and its state holds the final reward.
With this convention both learners converge to the expected discounted return

    Q(s,a) = E[ r_t + γ r_{t+1} + γ² r_{t+2} + ... | s_t = s, a_t = a ]

which `tests/convergence_test.cpp` verifies against a Monte Carlo estimate on small stochastic MDPs.
Put each reward on the link of the decision that earned it. Placing a reward on a *later* link delays it
by one step and changes the values: for example, putting the final reward on an extra terminal-state link
with a dummy action instead of on the last decision. When the only reward is the final one, this scales
the Q-values by γ and leaves the policy unchanged, but in general delaying rewards changes both the values
and the policy. Appending a link with zero reward changes nothing.

An episode cut short before the game ended (for example by a step limit) is not finished: pass
`false` as the learner's third argument, `learner(episode, policy, false)`. Its last link is then
not learnt as the end of the game, and only provides the next state for the link before it.

## Implementation in C++

The template class `policy` stores the learnt values of your states and actions.
The header provides a pair of template wrapper classes, `state` and `action`, which wrap around your own
state and action structures (the `state_trait` and `action_trait`). Those can be anything, provided that:
- they have an `operator==`
- they are hashable, with a specialisation of `std::hash`

Take care to make them unique: two different states whose traits compare equal are treated as one.
`examples/gridworld_header.hpp` shows how to write them.

You can also use your own state and action classes instead of the wrappers. They then need an `operator==`,
a specialisation of `relearn::hasher`, and states must expose their reward with a method `reward()`.

### Markov Decision Process

An MDP is a sequence of States  s<sub>t</sub> linked together by actions a<sub>t</sub>.
At the core of `relearn` is the `policy` which uses MDPs and **maps** them using the rewards `R` received along them.
Whilst multiple episodes may exist, and an episode may be iterated many times, updated, evaluated
and re-experienced, the actual process is done in class `policy`. An overview of an MDP is shown below.

![Markov Decision Process](https://github.com/alexge233/relearn/blob/master/images/mdp.png?raw=true)

The template parameters `state_trait` and `action_trait` are used to describe what a state and action are,
the wrappers are used only for indexing, hashing and policy calculation.

The basic use is to create episodes, denoted as `markov_chain`, which you populate as you see fit,
giving each step the reward it earned: any value, positive, negative or zero.

The use of a `markov_chain` defaults to an `std::deque<relearn::link<state_class,action_class>>`
where `relearn::link` is the struct encapsulating a state/action observation.

You may replace the container or link class if you want, but the learners require that:

- the episode has an `operator[]` taking the index of a link, and a `size()` method
- each link has the members `state` and `action`

So `std::deque` and `std::vector` work out of the box (`std::list` does not, as it has no `operator[]`).
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
Using `q_learning` or `q_probabilistic` you can then set/update the policy values by iterating your episodes.

# Dependencies

The library has **no external dependencies**!
However, your compiler **must support C++14**, for example:

- GCC 5 or higher
- Clang 3.4 or higher

These are the first releases with complete C++14 support. The continuous integration builds with GCC on Ubuntu.

__note__: I haven't tested with a Windows platform!

Building the examples and tests needs CMake 3.5 or higher.
The tests use [Catch2](https://github.com/catchorg/Catch2) v2, whose single header CMake downloads.

__note__: Serialization is optional, and needs Boost (see [enable serialization](#enable-serialization) below).

# Building

There is nothing to build! This is a header-only library, you only need to use the **relearn.hpp** header.
**However**, you may compile the examples in order to see how the library is used.

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
policies, states or actions: define `USING_BOOST_SERIALIZATION` (e.g., `-DUSING_BOOST_SERIALIZATION`)
before including `relearn.hpp`, and use CMake's `find_package` to find, include and link against `boost_serialization`.
Do bear in mind that the `state_trait` (e.g., your state **descriptor**) and the 
`action_trait` (e.g., your action **descriptor**) must **also be serializable**.
On how to achieve this, [have a look at this tutorial](http://www.boost.org/doc/libs/1_64_0/libs/serialization/doc/tutorial.html)
if this condition is not met, you will end up with compilation errors.

Because of the flexibility of boost serialization, you can save and load binary, text or xml archives.
Later versions of boost support smart pointers, so even if your descriptors are
`std::shared_ptr` or `std::unique_ptr` you can still save and load them.

# Examples

The folder `examples` holds two "Gridworld" examples:
- an offline one, which learns from random exploration: `examples/gridworld_offline.cpp` built as `ex_gridworld_offline`
- an online one, which learns after every episode: `examples/gridworld_online.cpp` built as `ex_gridworld_online`

and a collection of classic games, described in [Games](#games) below.

## basic usage

The basic way of using the library is the following:

1. create a class **state**, or use an existing class, structure, or POD which describes (in a *Markovian* sense) your state
2. create a class **action**, or use an existing class, structure, or POD which describes the action.
3. create an *episode*, which by default is an `std::deque<relearn::link<state,action>>`, and populate it with the steps taken and the reward each one earned.

At this point, depending on whether you are using an **online** or **offline** algorithm/approach, you have the following options:

4. keep creating episodes, and once you have finished, train the policy with all of them,
**or**
5. every time you create an episode, train your policy with it.

That choice is up to you, and almost always depends on the domain, system or problem you're trying to solve.
It is for this reason that the library has no implementation of `on_policy` or `off_policy` or `e_greedy`:
those are very simple algorithms, and are application-specific.
The games show one way to write them (`games::greedy` and `games::epsilon_greedy` in `examples/games/game.hpp`).

Take a look at the `gridworld` examples, which demonstrate two different ways of achieving the same task.
The `blackjack` example is different: Blackjack is a game of *chance*, where the same decision can win or lose
depending on the cards that follow. So it uses `q_probabilistic`, which learns the expected reward and the
probability of each transition from the games it plays.

## Gridworld

![gridworld image](https://github.com/alexge233/relearn/blob/master/images/gridworld.png?raw=true)

The pinnacle of simplicity when it comes to block/grid world toy problems, our agent resides in a 10x10 gridworld,
which is surrounded by blocks into which it can't move (black colour).
The agent starts at blue (x:1,y:8) and the target is the green (x:1,y:1).
The red blocks are fire/danger/a negative reward, and there is a rudimentary maze.
Moving into the target earns +1, moving into a fire block earns -1, and both end the episode.

Both versions learn with Q-learning, which learns the values of the best actions whatever actions were taken
while exploring. They differ in how they explore. The offline approach:

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

In this instance, the agent replays every episode 200 times: each pass moves the reward only one step back
towards the start, and random episodes can be long. The actual Q-Learning parameters are:
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
- if a good policy (a Q-value higher than zero, for a move that does not lead into a wall) exists, it follows it
- else if no good policy exists, it takes a random action
- repeats until a reward is found; the agent then **trains** its policies with the latest episode (10 passes)
- an episode that wanders for more than 1000 moves is cut short, and learnt with `terminated = false`
  (see [Reward convention](#reward-convention)), so that its last move is not learnt as the end of the game

The difference here, is that the actual exploration is instantly affected by what has already been learnt.
In comparison, the offline method is not affected, and may repeat the same sequences over and over again.
However, if the online version stops too early, there is no guarantee that the agent has learned the
ideal or optimal path to the goal, it could in fact be just a *mediocre* or *silly* path it has discovered.
This of course, is also a problem with offline, where the solution may never be discovered.

Other approaches exist (e.g., ε-greedy) where the agent may follow the policy,
but randomly choose to ignore it, in order to try and discover a better solution.
The games below use ε-greedy.

# Games

The folder `examples/games` holds classic reinforcement learning problems, each in its own header,
and `examples/<game>.cpp` learns each one (built as `ex_<game>`). Most come from Sutton & Barto [1];
FrozenLake, cart-pole and mountain car follow Gymnasium's [2] rules and physics, so agents can later be
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

`games::play_online` plays an episode and learns after every step instead, as Sutton & Barto's
TD algorithms do. SARSA needs it with a large learning rate (see [docs/sarsa.md](docs/sarsa.md)):

```cpp
auto r = games::play_online(game, agent, learner, memory, 1000);
```

| Game | Source | Learnt with | What it shows |
|---|---|---|---|
| `bandit` | S&B 2.3 | `q_probabilistic` | Exploration vs exploitation; greedy vs ε-greedy (Figure 2.2) |
| `blackjack` | S&B 5.1 | `q_probabilistic` | A random game; the learnt strategy matches Figure 5.2 |
| `cliff_walking` | S&B 6.6 | `q_learning`, `sarsa`, `expected_sarsa` | On-policy vs off-policy: Q-learning learns the path along the cliff edge, SARSA a safer one, and both SARSAs earn more while exploring |
| `windy_gridworld` | S&B 6.5, Ex. 6.10 | both | Steady wind: the 15-move path. Random wind: `q_probabilistic`, and `q_learning` with a learning rate that decays per pair, find the best policy; `q_learning` with α = 0.5 does not |
| `random_walk` | S&B 6.2, 7.1 | `q_probabilistic` | Prediction: learnt values next to the exact ones |
| `maximization_bias` | S&B 6.7 | `q_learning` | Q-learning's overestimation (for Double Q-learning) |
| `gamblers_problem` | S&B 4.3 | `q_probabilistic` | Values that match the exact solution |
| `dyna_maze` | S&B 8.1–8.3 | `q_learning` | Learning a maze; mazes that change (for Dyna-Q, Dyna-Q+) |
| `tic_tac_toe` | S&B 1.5 | `q_learning` | Against a random or a perfect (minimax) opponent |
| `frozen_lake` | Gymnasium | `q_probabilistic` | Slippery ice; the learnt 4x4 policy is optimal |
| `cart_pole` | Gymnasium | `q_learning` | A continuous state, cut into boxes (`games::discretised`) |
| `mountain_car` | Gymnasium | `q_learning` | A continuous state; exploring by optimistic initial values |

`tests/games_test.cpp` tests the rules of every game and that the learners reach the known answer:
exact solutions (`tests/solvers.hpp`: value iteration, exact blackjack, breadth-first search) or the
optimal results given by the book.

# Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first:
it describes the code style, and how changes are tested.
AI-generated contributions are welcome, as long as they have been reviewed and approved by a human.

# TODO

1. Double Q-learning (`maximization_bias`)
2. Dyna-Q and Dyna-Q+ (`dyna_maze`)
3. the R-Learning continuous algorithm
4. eligibility traces

# References

1. Sutton, R.S. and Barto, A.G., 2018. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 2nd edition. Cambridge, MA: MIT Press.
   Section, example and figure numbers above refer to this edition.
2. [Gymnasium](https://gymnasium.farama.org), the maintained successor of OpenAI Gym.
