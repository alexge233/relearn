# WARNING - Work in Progress
# relearn 0.1.0

A [Reinforcement Learning][Sutton, R.S. and Barto, A.G., 1998. Reinforcement learning: An introduction (Vol. 1, No. 1). Cambridge: MIT press] *header-only* itemplate library for C++11/14.
Minimal and simple to use, for a variety of scenarios.
It is based on [Sutton and Barto's book](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html) 
and implements *some* of the functionality described in it.

## Reinforcement Learning and Markov Decision Process (MDP)

An MDP is a sequence of States  s<sub>t</sub> linked together by actions a<sub>t</sub>.
At the core of `relearn` is the `episode`; it encapsulates around an MDP and owns it.
Whilst multiple episodes may exist, an episode may be iterated many times, updated, evaluated
and re-experienced.

![MDP](http://g.gravizo.com/g?
[*] --> State1;
State1 : t = 1;
State1 --> State2 : action 1;
State2: t = 2;
State2 --> State3 : action 1;
State3: t = 3;
State1 --> State4: action 2;
State4 : t =2;
State1 --> State5 : action 3;
State5 : t = 2;
State2 --> State6 : action 2;
State6 : t = 3;
State2 --> State7 : action 3;
State7 : t = 3;
State5 --> State8 : action 1;
State8 : t = 3;
State3 --> State9: action 1;
State9 : t = terminal;
State5 --> State10 : action 2;
State10 : t = 3;
State8 --> State11 : action 1;
State10 --> State11 : action 1;
State11 : t = 4;
State4 --> State8 : action 1;
State4 --> State7 : action2;
@enduml)
        
At the heart of episodic learning we implement [Q-learning](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html) 
and [SARSA](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node64.html)
and for continous learning we implement [R-Learning](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node67.html).

 
