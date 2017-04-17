# relearn 0.1.0

A [Reinforcement Learning][1] *header-only* template library for C++14.
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

Furthermore, the use of a `markov_chain` defaults to an `std::deque<relearn::link<state_class,action_class>>`
where `relearn::link` is simply the struct encapsulating a state/action observation.
You may replace the container or link class if you want, but the `q_learning` structure
requires that:

- an episode (a `markov_chain`) can be iterated
- the iterator can be used with `std::next`
- there exists an `::end()` method to denote the last item in the episode

Basically most *STL* containers should work right out of the box.
Finally, the library uses by default `double` for storing and updating Q values,
but you can play around with it if needed.

## Markov Decision Process

An MDP is a sequence of States  s<sub>t</sub> linked together by actions a<sub>t</sub>.
At the core of `relearn` is the `policy` which uses MDPs and **maps** them using the final state reward `R`.
Whilst multiple episodes may exist, and an episode may be iterated many times, updated, evaluated
and re-experienced, the actual process is done in class `policy`. An overview of an MDP is shown below.

![Markov Decision Process](https://github.com/alexge233/relearn/blob/master/mdp.png?raw=true)

At the heart of episodic learning I've implemented [Q-learning](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html) 
and for continous learning I plan to implement [R-Learning](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node67.html).

# Status

Currently the supported Algorithm is a *Deterministic Q-Learning*. 
I'm working on a *Non-Deterministic* (Probabilistic) Q-Learning, and after that I'll Implement *Continuous* R-Learning. 

As of 0.1.0 the library is **WORK IN PROGRESS**

# Building

There is nothing to build! This is a header-only library, you only need to use the **relearn.hpp** header.
However, you may compile the examples (currently only `gridworld` will build) in order to see how the library is implemented.

# Examples

There is a folder `examples` which I'm populating with examples, starting from your typical *gridworld* problem, 
and then moving on to a *blackjack* program.

## Gridworld

The pinacle of simplicity when it comes to block/grid world toy problems, our agent resides in a 10x10 gridworld,
which is surrounded by blocks into which he can't move (black colour).
The agent starts at blue (x:1,y:8) and the target is the green (x:1,y:1).
The red blocks are fire/danger/a negative reward, and there is a rudimentary maze.

## Blackjack

A simplified attempt, where one player uses classic probabilities, the dealer (house) simply draws until 17,
and the adaptive agent uses non-deterministic Q-learning in order to play as best as possible.

## TODO

1. complete the blackjack example
2. write unit tests 
3. complete the crazyflie example

[1]: Sutton, R.S. and Barto, A.G., 1998. Reinforcement learning: An introduction (Vol. 1, No. 1). Cambridge: MIT press
