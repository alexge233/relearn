![RELEARN](https://github.com/alexge233/relearn/blob/master/images/relearn.png?raw=true)

A Reinforcement Learning *header-only* template library for C++14.
Minimal and simple to use, for a variety of scenarios.
It is based on [Sutton and Barto's book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html) 
and implements *some* of the functionality described in it.

Currently the library implements Q-Learning for deterministic systems, as well as non-deterministic systems.

The Q-Learning (for both offline and online policy update rule) is:

![Q(s_t,a_) = Q(s_t,a_t) + alpha * (R_t + gamma * max(Q_{s+t},a_t) - Q(s_t,a_t))](https://github.com/alexge233/relearn/blob/master/images/q_learning.png?raw=true)

The policy decision is entirely up to you to decide; in the case of a deterministic system,
usually the options are to either stay on policy:

![a_t = Π(s_t)](https://github.com/alexge233/relearn/blob/master/images/deterministic_policy.png?raw=true)

or if using online explorative-greedy (ε-Greedy) assign some *decaying probability* on using the policy
or taking a random action:

![P_{s_t}^{Π}(a_t)](https://github.com/alexge233/relearn/blob/master/images/stochastic_policy.png?raw=true)

For Q-Learning in a non-deterministic system (e.g., when the certainty of transitioning from one state to another
is not constant) we use a transition function:

![f = f(s_t,s_{t+1})](https://github.com/alexge233/relearn/blob/master/images/transition_func.png?raw=true)

That transition function, suggests that from state `s_t` if taking action `a_t` the agent ends up in state `s_{t+1}`.
The assigned probability of that transition must be **observed** after being taken:

![P(s_t,a_t)(f) = Σ P(s_t,a_t)(s_{t+1})](https://github.com/alexge233/relearn/blob/master/images/transition_prob.png?raw=true)

This probability quantifies the *likelihood* of the transition `s_t → s_{t+1}` for action `a_t`.
Similarly, the *Expected reward* is:

![E(s_t,s_{t+1}) = Σ f(s_t,s_{t+1}) P(s_t,a_t)(f)](https://github.com/alexge233/relearn/blob/master/images/expected_reward.png?raw=true)

This simply states that the reward is equivalent to the reward of the ending-up state `s_{t+1}` affected by the transition probability.
Therefore, the update rule for probabilistic systems is:

![Q(s_t,a_t) = E(r_{s_{t+1}}) + gamma * Σ maxQ(s_{t+1},a_t)P(s_t,a_t)(s_{t+1})](https://github.com/alexge233/relearn/blob/master/images/q_probabilistic.png?raw=true)

### Reward convention

In an episode, each link `(s_t, a_t)` carries in `s_t.reward()` the reward received *after* taking `a_t` in `s_t`.
This is the `r_t` above, which Sutton & Barto (2nd ed.) write as `R_{t+1}`: the same reward, a different index.
The last link of an episode is the final decision, and its state holds the final reward.
With this convention both learners converge to the expected discounted return

    Q(s,a) = E[ r_t + γ r_{t+1} + γ² r_{t+2} + ... | s_t = s, a_t = a ]

which `tests/convergence_test.cpp` verifies against a Monte Carlo estimate on small stochastic MDPs.
Put each reward on the link of the decision that earned it. Placing a reward on a *later* link delays it
by one step and changes the values: for example, putting the final reward on an extra terminal-state link
with a dummy action (as the gridworld example does) instead of on the last decision. When the only reward is
the final one, this scales the Q-values by γ and leaves the policy unchanged, but in general delaying rewards
changes both the values and the policy. Appending a link with zero reward changes nothing.

An episode cut short before the game ended (for example by a step limit) is not finished: pass
`false` as the learner's third argument, `learner(episode, policy, false)`. Its last link is then
not learnt as the end of the game, and only provides the next state for the link before it.

## Implementation in C++

The template class `policy` allows the use of your own `state` and `action` classes,
although a pair of template wrapper classs `state` and `action` is provided in the header.
You can create your own `state` and `action` classes and use them, but they must satisfy the below criteria:
- terminal states must have a reward exposed by a method `::reward()`
- states and actions must be hashable

The classes `state` and `action` provided in the header wrap around your own state and action structures,
which can be anything, provided they are hashable.
Take care to make them unique, otherwise the result may be undefined behaviour.

### Markov Decision Process

An MDP is a sequence of States  s<sub>t</sub> linked together by actions a<sub>t</sub>.
At the core of `relearn` is the `policy` which uses MDPs and **maps** them using the final state reward `R`.
Whilst multiple episodes may exist, and an episode may be iterated many times, updated, evaluated
and re-experienced, the actual process is done in class `policy`. An overview of an MDP is shown below.

![Markov Decision Process](https://github.com/alexge233/relearn/blob/master/images/mdp.png?raw=true)

The template parameters `state_trait` and `action_trait` are used to describe what a state and action are,
the wrappers are used only for indexing, hashing and policy calculation.

The basic use is to create episodes, denoted as `markov_chain` which you populate as you see fit,
and then reward with either a negative (-1), neutral (0) or positive (+1) value.

The use of a `markov_chain` defaults to an `std::deque<relearn::link<state_class,action_class>>`
where `relearn::link` is the struct encapsulating a state/action observation.

You may replace the container or link class if you want, but the `q_learning` structure
requires that:

- an episode (a `markov_chain`) can be iterated
- there exists an `::size()` method to denote the size of the episode

Basically most *STL* containers should work right out of the box.
The library uses by default `double` for storing and updating Q values,
but you can play around with it if needed.

The class that mostly interests us is `policy` which uses your states and action,
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
Using `q_learning` you can then set/update the policy values by iterating your episodes.

# Dependencies

There are **no external dependencies**!
However, your compiler **must support C++14**, so you will need:

- gcc 4.8.4 or higher
- clang 3.3 or higher

__note__: I haven't tested with a Windows platform!

__note__: If you need serialization, current `master` branch has a flag `USING_BOOST_SERIALIZATION`,
which when set to `ON` will enable boost serialization, provided that your states and actions (template parameters `state_trait` and `action_trait` are indeed serializable). 
In this case, you need to use CMake's `find_package` to properly find, include and link against `boost_serialization`.

# Building

There is nothing to build! This is a header-only library, you only need to use the **relearn.hpp** header.
**However**, you may compile the examples in order to see how the library is implemented.

To do that, simply:

```bash
mkdir build
cd build
cmake ..
make
```

### build examples

Simply do the following:

```bash
cmake .. -DBUILD_EXAMPLES=On
make
```

### build tests

```bash
cmake .. -DBUILD_TESTS=On
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
cmake .. -DUSING_BOOST_SERIALIZATION=On -DBUILD_EXAMPLES=On
```

You can also set this flag for your **own** project, if you wish to save and load
policies, states or actions.
Do bear in mind that the `state_trait` (e.g., your state **descriptor**) and the 
`action_trait` (e.g., your action **descriptor**) must **also be serializable**.
On how to achieve this, [have a look at this tutorial](http://www.boost.org/doc/libs/1_64_0/libs/serialization/doc/tutorial.html)
if this condition is not met, you will end up with compilation errors.

Because of the flexibility of boost serialization, you can save and load binary, text or xml archives.
Later versions of boost support smart pointers, so even if your descriptors are
`std::shared_ptr` or `std::unique_ptr` you can still save and load them.

# Examples

The folder `examples` holds two "Gridworld" examples:
- an offline on-policy algorithm: `examples/gridworld_offline.cpp` built as `ex_gridworld_offline`
- an online on-policy algorithm: `examples/gridworld_online.cpp` built as `ex_gridworld_online`

and a collection of classic games, described in [Games](#games) below.

## basic usage

The basic way of using the library is the following:

1. create a class **state**, or use an existing class, structure, or PDT which describes (in a *Markovian* sense) your state
2. create a class **action**, or use an existing class, structure, or PDT which describes the action.
3. create an *episode* which by default is an `std::deque<relearn::link<state,action>>` which you populate, and then reward.

At this point, depending on wether you are using an **online** or **offline** algorith/approach, you have the following options:

4. keep creating episodes, obtain a reward for the last/terminal state, and once you have finished, train the policy will all of them,
**or**
5. every time you create an episode, obtain the reward, then you can train your policy with it.

That choice is up to you, and almost always depends on the domain, system or problem you're trying to solve.
It is for this reason that there is no implementation of `on_policy` or `off_policy` or `e_greedy`, 
those are very simple algorithms, and are application-specific.

Take a look at the `gridworld` examples, which demonstrate two different ways of achieving the same task.
The `blackjack` example is different: because playing Blackjack is a *probability* task, we can't use
a *deterministic* approach, rather we use a *probabilistic* approach, in which case

1. we have to take an offline approach (we don't know the transition from one state to another until we've experienced it)
2. we have to train on probabilities of transitioning (e.g., non-deterministic)

## Gridworld

![gridworld image](https://github.com/alexge233/relearn/blob/master/images/gridworld.png?raw=true)

The pinacle of simplicity when it comes to block/grid world toy problems, our agent resides in a 10x10 gridworld,
which is surrounded by blocks into which he can't move (black colour).
The agent starts at blue (x:1,y:8) and the target is the green (x:1,y:1).
The red blocks are fire/danger/a negative reward, and there is a rudimentary maze.

There are two versions of the Gridworld, the offline approach:

- first the agent randomly explores until it can find the positive reward (+1.0) grid block
- then it updates its policies
- finally it follows the best policy learnt

And the online approach:

- the agent randomly explores one episode
- then it updates its policies
- then it tries again, this time going after known policies
- it only falls back to random when there does not exist a *positive* best action
- the entire process is repeated until the goal is discovered.

The actual gridworld is saved in a textfile `gridworld.txt` (feel free to change it).
The example `examples/gridworld_header.hpp` provides the minimal code to demonstrate this staged approach.
The two files:

- `examples/gridworld_offline.cpp`
- `examples/gridworld_online.cpp`

have the different versions of how this task can be solved.

Once we have loaded the world (using function `populate`) we set the start at x:1, y:8 and then
begin the exploration.

### offline q-learning

The offline exploration runs in an inifinite until the grid block with a **positive** reward is found.
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
- sets as root state the starting gridblock
- randomly picks a direction (see struct `rand_direction` for more)
- repeats this until either (a) a negative reward has been found (e.g., stepped into a fire block), or (b) the goal block is discovered

Each time the `expore` method is called in the main exploration for loop,
an episode is acquired, which is then appended in the list of experienced episodes `std::vector<std::deque<relearn::link<state,action>>> episodes`.

Once the agent has found the target gridblock, he updates his policy memory using those episodes:

```cpp
auto learner = relearn::q_learning<state,action>{0.9, 0.1};
for (int k = 0; k < 10; k++) {
    for (auto episode : episodes) {
        learner(episode, policies);
    }
}
```

In this instance, the agent will run ten iterations for each episode, and the actual Q-Learning parameters are:
- alpha (learning rate) 0.9
- gamma (discount) 0.1

Finally, once updating policies has finished, the agent proceeds to follow (once) the best action policy:

```cpp
on_policy(w, policies, start);
```

The result is that the agent will go straight from x:1, y:8 to x:1, y:1 avoiding any fire blocks
and unnecessarily searching the gridworld.

This is a __deterministic__ scenario, because the agent knows at any given moment, which action he is taking,
and to __which__ state that action will lead to.

### online q-learning

The online exploration is somewhat different, because the `explore` method does the following:

- creates a new episode
- sets the root state
- if a good policy (Q-value higher than zero and a valid action pointer) exist, it follows them
- else if no good policy (or action pointer) exist, it takes a random action
- repeats until a reward is found, it then **trains** the policies with the latest episode

The difference here, is that the actual exploration is instantly affected by what has already be learnt.
In comparison, the offline method is not affected, and may repeat the same sequences over and over again.
However, if the online version stops too early, there is no guarantee that the agent has learned the
ideal or optimal path to the goal, it could in fact be just a *mediocre* or *silly* path it has discovered.
This of course, is also a problem with offline, where the solution may never be discovered.

Other more complex algorithms exist (e.g., e-greedy) where the agent may follow the policy,
but randomly chose to ignore it, in order to try and discover a better solution.

# Games

The folder `examples/games` holds classic reinforcement learning problems, each in its own header,
and `examples/<game>.cpp` learns each one (built as `ex_<game>`). Most come from Sutton & Barto [1];
FrozenLake, cart-pole and mountain car follow Gymnasium's rules and physics, so agents can later be
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

| Game | Source | Learnt with | What it shows |
|---|---|---|---|
| `bandit` | S&B 2.3 | `q_probabilistic` | Exploration vs exploitation; greedy vs ε-greedy (Figure 2.2) |
| `blackjack` | S&B 5.1 | `q_probabilistic` | A random game; the learnt strategy matches Figure 5.2 |
| `cliff_walking` | S&B 6.6 | `q_learning` | The 13-move path along the cliff edge |
| `windy_gridworld` | S&B 6.5, Ex. 6.10 | both | Steady wind: the 15-move path. Random wind: `q_probabilistic` finds the best policy, `q_learning` with a fixed learning rate does not |
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

# TODO

1. do the R-Learning continous algorithm
2. add eligibility traces (decay)

[1]: Sutton, R.S. and Barto, A.G., 1998. Reinforcement learning: An introduction (Vol. 1, No. 1). Cambridge: MIT press
