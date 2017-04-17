# relearn 0.1.0

A Reinforcement Learning *header-only* template library for C++14.
Minimal and simple to use, for a variety of scenarios.
It is based on [Sutton and Barto's book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html) 
and implements *some* of the functionality described in it.

Currently the library implements Q-Learning for deterministic systems, as shown below:

!["Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \bigg[R_{t+1} + \gamma * \underset{a}{max}Q(s_{t+1}, a) - Q(s_t, a_t) \bigg]"](http://www.sciweavers.org/tex2img.php?eq=Q%28s_t%2Ca_t%29%20%20%20%20%5Cleftarrow%20Q%28s_t%2Ca_t%29%20%2B%20%20%5Calpha%20%5Cbigg%5BR_%7Bt%2B1%7D%20%2B%20%20%5Cgamma%20%2A%20%5Cunderset%7Ba%7D%7Bmax%7DQ%28s_%7Bt%2B1%7D%2C%20a%29%20-%20Q%28s_t%2C%20a_t%29%20%5Cbigg%5D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

The template class `policy` allows the use of your own `state` and `action` classes,
although a pair of template wrapper classs `state` and `action` is provided in the header.
The requirements are:
- terminal states must have a reward
- states and actions must be hashable


## Markov Decision Process

An MDP is a sequence of States  s<sub>t</sub> linked together by actions a<sub>t</sub>.
At the core of `relearn` is the `policy` which uses MDPs and **maps** them using the final state reward `R`.
Whilst multiple episodes may exist, and an episode may be iterated many times, updated, evaluated
and re-experienced, the actual process is done in class `policy`. An overview of an MDP is shown below.

![Markov Decision Process](https://github.com/alexge233/relearn/blob/master/mdp.png?raw=true)

At the heart of episodic learning I've implemented [Q-learning](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html) 
(for continous learning I plan to implement [R-Learning](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node67.html). in the near future).

The use of a `markov_chain` defaults to an `std::deque<relearn::link<state_class,action_class>>`
where `relearn::link` is the struct encapsulating a state/action observation.
You may replace the container or link class if you want, but the `q_learning` structure
requires that:

- an episode (a `markov_chain`) can be iterated
- the iterator can be used with `std::next`
- there exists an `::end()` method to denote the last item in the episode

Basically most *STL* containers should work right out of the box.
Finally, the library uses by default `double` for storing and updating Q values,
but you can play around with it if needed.

# Status

Currently the supported Algorithm is a *Deterministic Q-Learning*. 
I'm working on a *Non-Deterministic* (Probabilistic) Q-Learning, and after that I'll Implement *Continuous* R-Learning. 

As of 0.1.0 the library is **WORK IN PROGRESS**

# Dependencies

There are **no external dependencies**!
However, your compiler **must support C++14**, so you will need:

- gcc 4.8.4 or higher
- clang 3.3 or higher

__note__ I haven't tested with a Windows platform!

# Building

There is nothing to build! This is a header-only library, you only need to use the **relearn.hpp** header.
However, you may compile the examples (currently only `gridworld` will build) in order to see how the library is implemented.

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

## Gridworld

![gridworld image](https://github.com/alexge233/relearn/blob/master/gridworld.png?raw=true)

The pinacle of simplicity when it comes to block/grid world toy problems, our agent resides in a 10x10 gridworld,
which is surrounded by blocks into which he can't move (black colour).
The agent starts at blue (x:1,y:8) and the target is the green (x:1,y:1).
The red blocks are fire/danger/a negative reward, and there is a rudimentary maze.

This example uses a staged approach:

- first the agent randomly explores until it can find the positive reward (+1.0) grid block
- then it updates its policies
- finally it follows the best policy learnt

The actual gridworld is saved in a textfile `gridworld.txt` (feel free to change it).
The example `src/gridworld.cpp` provides the minimal code to demonstrate this staged approach.

Once we have loaded the world (using function `populate`) we set the start at x:1, y:8 and then
begin the exploration.

The exploration runs in an inifinite loop in `main` until one criterion is satisfied: the grid block with a **positive** reward is found.
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

## TODO

1. complete the blackjack example
2. do the Q-Learning non-deterministic
3. complete the crazyflie example
4. do the R-Learning continous algorithm

[1]: Sutton, R.S. and Barto, A.G., 1998. Reinforcement learning: An introduction (Vol. 1, No. 1). Cambridge: MIT press
