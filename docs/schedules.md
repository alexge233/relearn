# Schedules: learning and exploring less as you go

Two numbers decide how an agent learns, and neither has to stay fixed:

- the **learning rate** `α`: how far each new experience moves a value,
- the **exploration rate** `ε`: how often the agent tries a random move instead of the best one.

At the start the agent knows nothing, so it should learn fast and explore a lot.
Later it knows more, so it should learn more carefully and explore less.
A *schedule* says how a number changes as learning goes on.
This document explains the schedules relearn provides, when each one helps,
and what the theory of Sutton and Barto [1] asks of them.

## Schedules in relearn

A schedule is called with a count `k = 1, 2, 3, ...` and returns a value.
For the learning rate, `k` is the number of times a state/action pair has been updated;
for the exploration rate, `k` is usually the episode.

| schedule | value at step `k` | typical use |
|---|---|---|
| `polynomial_decay(initial, exponent)` | `initial / k^exponent` | learning rate, ε |
| `exponential_decay(initial, rate, minimum)` | `max(minimum, initial · rate^(k-1))` | ε |
| `linear_decay(initial, last, steps)` | from `initial` to `last` in a straight line over `steps` steps, then `last` | ε |

```math
\text{polynomial:}\; v(k) = \frac{v_0}{k^{\omega}} \qquad
\text{exponential:}\; v(k) = \max\left(v_{min},\; v_0\, \lambda^{k-1}\right) \qquad
\text{linear:}\; v(k) = v_0 + (v_1 - v_0)\,\min\left(\frac{k-1}{n}, 1\right)
```

They are plain functions of `k`, with no dependencies, so they can drive anything that should change over time.

## The learning rate

With a constant `α`, every update moves a value a fixed fraction of the way towards its new target.
That keeps the agent adapting, but in a random environment the values never settle:
each lucky or unlucky outcome moves them again.

For the values to converge, the learning rates `α_n` used on the `n`-th update of each
state/action pair must satisfy two conditions (Sutton and Barto, equation 2.7):

```math
\sum_{n=1}^{\infty} \alpha_n = \infty \qquad \text{and} \qquad \sum_{n=1}^{\infty} \alpha_n^2 < \infty
```

The first says the steps are large enough to get anywhere from any starting value;
the second says they shrink fast enough for the noise to average out.
`polynomial_decay` meets both when `0.5 < exponent ≤ 1`.

Two details matter in practice:

- **The rate must decay for each pair, by its own number of updates.** A pair visited rarely
  should not have its rate cut because *other* pairs were visited. That is why the learners
  count the updates of each pair, and pass that count `n` to the schedule.
- **`α = 1/n` is often too slow.** With `α = 1/n` each value is the plain average of all
  its targets. But the targets of Q-learning and SARSA are built from other values,
  `r + γ Q(s', a')`, which are wrong at first. `1/n` gives those early, wrong targets as much
  weight as the later, better ones, so learning crawls. A slower decay, such as `1/n^0.6`,
  forgets the early targets faster and still meets both conditions.

### Pseudocode

```
a learning rate per state/action pair

parameters: a schedule α(n), e.g. α(n) = 1 / n^0.6

start with n(s, a) = 0 for every state s and action a

for each update of Q(s, a) towards a target:
    n(s, a) ← n(s, a) + 1
    Q(s, a) ← Q(s, a) + α(n(s, a)) [ target - Q(s, a) ]
```

### In relearn

`q_learning`, `sarsa` and `expected_sarsa` have an optional `learning_rate`.
When it is set, the `n`-th update of each pair uses `learning_rate(n)` instead of `alpha`:

```cpp
relearn::q_learning<state, action> learner;
learner.gamma = 1.0;
learner.learning_rate = relearn::polynomial_decay<>(1.0, 0.6);   // α = 1/n^0.6
```

The learner counts the updates of each pair in `learner.visits`.
If `learning_rate` is not set, `alpha` is used for every update, as before.

### Example: Windy Gridworld in random wind

In random wind (Sutton & Barto, Exercise 6.10) the same move can end in different cells.
`examples/windy_gridworld.cpp` trains Q-learning for 5000 episodes, with different learning rates.
Average moves of the greedy policy from the start (the best policy needs 20.6):

| learning rate | moves |
|---|---|
| `α = 0.5`, constant | 37.3 |
| `α = 1/n^0.6`, per pair | 20.8 |
| `q_probabilistic` (averages by design) | 20.5 |

Over 10 runs with other seeds (2000 and 10000 episodes, each policy measured over 1000 greedy episodes):

| learning rate | 2000 episodes | 10000 episodes |
|---|---|---|
| `α = 0.5`, constant | 499 | 321 |
| `α = 0.1`, constant | 21.3 | 20.7 |
| `α = 1/n`, per pair | 859 | 796 |
| `α = 1/n^0.6`, per pair | 217 | 20.7 |
| `α = 0.5/√n`, per pair | 120 | 20.7 |

A large constant rate keeps chasing the latest gust of wind; `1/n` decays too fast to get
anywhere; `1/n^0.6` and `0.5/√n` reach the best policy. A small constant rate (`0.1`) also does
well here: in practice a small constant is often good enough, but only a decaying rate converges.
`tests/games_test.cpp` checks, by evaluating the learnt policy exactly, that `1/n^0.6`
learns the best policy (it did in 10 runs out of 10).

## The exploration rate

With a constant `ε`, the agent explores forever.

- **Q-learning** is off-policy: it learns the values of the best policy, `q*`, however much it explores.
  Exploring forever only costs it reward while it plays.
- **SARSA and Expected SARSA** are on-policy: they learn the values of the policy they follow,
  exploration included. With a constant `ε` they learn `q_ε`, the values of the ε-greedy policy,
  not `q*`.

To learn `q*`, an on-policy learner must explore less and less, but never stop for good.
Sutton and Barto (Section 6.4) state it as: every state/action pair keeps being visited
(infinitely often), and the policy becomes greedy in the limit (often called *GLIE*).

`ε = 1/k` in episode `k` makes the policy greedy in the limit, and since `Σ 1/k = ∞`
the agent never stops exploring altogether. But that alone does not guarantee that *every* pair
keeps being visited: a pair that the agent only reaches by exploring twice in a row is visited with
a probability of about `1/k²`, and `Σ 1/k²` is finite, so it may stop being visited.
An exploration rate per state does guarantee it: `ε(s) = c / n(s)`, where `n(s)` is the number of
visits of state `s` (Singh, Jaakkola, Littman and Szepesvári, 2000 [2]).
In practice `ε = 1/k` per episode is common, and works well when every pair is one exploratory move
away from the states the agent visits anyway, as in the corridor below.

`exponential_decay` with `minimum = 0` does not: `Σ λ^k` is finite, so after a while the agent
practically stops exploring, and pairs it has not tried enough are never learnt.
With `minimum > 0` it keeps exploring, but never becomes greedy.
Both are still common and useful in practice; they just do not meet the theory.

### Pseudocode

```
SARSA with exploration that fades out

parameters: learning rates α(n), exploration rates ε(k), e.g. ε(k) = 1/k

for each episode k = 1, 2, 3, ...:
    ε ← ε(k)
    s ← the start state
    a ← an action for s, chosen ε-greedy on Q
    repeat until the episode ends:
        take action a, and observe the reward r and the next state s'
        n(s, a) ← n(s, a) + 1
        if s' ends the episode:
            Q(s, a) ← Q(s, a) + α(n(s, a)) [ r - Q(s, a) ]
        else:
            a' ← an action for s', chosen ε-greedy on Q
            Q(s, a) ← Q(s, a) + α(n(s, a)) [ r + γ Q(s', a') - Q(s, a) ]
            s ← s'
            a ← a'
```

### In relearn

relearn does not choose actions, so the exploration rate lives in your agent:

```cpp
relearn::polynomial_decay<> fade(1.0, 1.0);          // ε = 1/k
for (std::size_t k = 1; k <= episodes; k++) {
    double epsilon = fade(k);
    // play episode k, exploring with probability epsilon
}
```

For Expected SARSA, the policy `π` you give it must use the same, current `ε`.

### Example: a slippery corridor

`tests/sarsa_test.cpp` learns a small slippery corridor, whose exact values are known,
with SARSA and `α = 1/n^0.6` per pair, for 20000 episodes (worst errors over 10 runs):

| exploration | worst error to `q_ε` | worst error to `q*` |
|---|---|---|
| `ε = 0.1`, constant | 0.07 (all moves) | 0.27 (all moves) |
| `ε = 1/k`, fading | | 0.045 (moving right, the best move) |

With a constant `ε`, SARSA learns that exploring has a cost: from the first cell, moving right is
worth -4.31 under the ε-greedy policy, against -4.06 for the best policy.
With `ε = 1/k`, it learns the best values of the moves it comes to take. In the corridor, every
pair is one exploratory move away from the path the agent takes, so `ε = 1/k` keeps visiting all of them.
The moves it takes less and less often learn slowly (errors up to 1.2 after 20000 episodes):
that is the price of exploring less.

## Summary

- Use a **per-pair decaying learning rate**, `polynomial_decay` with an exponent between 0.5 and 1
  (e.g. `1/n^0.6`), when the environment is random and the values should converge.
- A **small constant learning rate** is simpler and often good enough, but never fully settles.
- Use a **fading exploration rate**, e.g. `ε = 1/k`, when an on-policy learner (SARSA, Expected SARSA)
  should learn the best policy rather than the ε-greedy one.

## References

1. Sutton, R.S. and Barto, A.G., 2018. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 2nd edition. Cambridge, MA: MIT Press.
   Section 2.4 (sample averages), Section 2.5 and equation 2.7 (conditions on the learning rate),
   Section 6.4 (SARSA's convergence), Exercise 6.10 (random wind).
2. Singh, S., Jaakkola, T., Littman, M.L. and Szepesvári, C., 2000. Convergence results for single-step
   on-policy reinforcement-learning algorithms. *Machine Learning*, 38(3), pp. 287-308.
