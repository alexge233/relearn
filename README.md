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

# Examples

There is a folder `examples` which I'm populating with examples, starting from your typical *gridworld* problem, 
and then moving on to a *blackjack* program.
Currently there is a classical "Gridworld" example, with two versions:
- an offline on-policy algorithm: `examples/gridworld_offline.cpp` built as `ex_gridworld_offline`
- an online on-policy algorithm: `examples/gridworld_online.cpp` built as `ex_gridworld_online`

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
The example `src/gridworld.cpp` provides the minimal code to demonstrate this staged approach.

Once we have loaded the world (using function `populate`) we set the start at x:1, y:8 and then
begin the exploration.

The exploration runs in an inifinite until the grid block with a **positive** reward is found.
Until that happens, the agent takes a *stochastic* (e.g., random) approach and searches the gridworld.
The function:

```cpp
template <typename S, 
          typename A>
std::deque<relearn::link<S,A>> explore(const world & w,
                                       std::mt19937 & gen,
                                       grid start);
```

does the following:

- creates a new episode (e.g., `relearn::markov_chain`)
- sets as root state the starting gridblock x:1, y:8
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

## Blackjack

A simplified attempt, where one player uses classic probabilities, the dealer (house) simply draws until 17,
and the adaptive agent uses non-deterministic Q-learning in order to play as best as possible.

The `state` is very simple: a `hand` which is described by the value (min value and max value, depending on the cards held).
The agent ignores the dealer's hand since that would increase the state space,
as well as the label or symbol of the cards held (feel free to change this, simply adapt the "hash" function of `hand`).

This example takes a lot of time to run, as the agent maps the transitional probabilities,
using the observations from playing multiple games.

## TODO

1. implement the `boost_serialization` with internal header
2. do the R-Learning continous algorithm

[1]: Sutton, R.S. and Barto, A.G., 1998. Reinforcement learning: An introduction (Vol. 1, No. 1). Cambridge: MIT press
