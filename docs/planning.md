# Planning with a model: trajectory sampling, real-time dynamic programming, heuristic search and rollouts

Dynamic programming (have a look at [dynamic_programming.md](dynamic_programming.md)) plans with a model of the world,
`p(s', r | s, a)`, and sweeps over *every* state, again and again. On a large problem most of those states never matter:
the agent starts in a few of them, and a good policy never goes near most of the others. Sutton and Barto's Chapter 8 [1]
gives several ways of spending the computation where it counts, and relearn has four of them, in
`relearn/planning.hpp`. All four take the same model as `dynamic_programming`, a `relearn::finite_mdp`:

- **trajectory sampling** (Section 8.6): expected updates of the action values, spread uniformly over every pair, or along
  trajectories that follow the current policy
- **real-time dynamic programming** (Section 8.7): value iteration, but only on the states met along greedy trajectories
- **heuristic search** (Section 8.9): look a few moves ahead from the state the agent is in, through every outcome
- **rollout algorithms** (Section 8.10): estimate each action by simulating episodes that then follow a rollout policy

The first two learn values; the last two decide what to do in the state the agent is in, when it is there, and keep
nothing. Monte Carlo Tree Search (Section 8.11, [mcts.md](mcts.md)) is the rollout algorithm that grows a tree as it goes.

## Trajectory sampling

Every update here is an *expected* update of one state/action pair, from the model:

```math
Q(s, a) \leftarrow \sum_{s', r} p(s', r \mid s, a) \left( r + \gamma \max_{a'} Q(s', a') \right)
```

The question is which pairs to update. The *uniform* distribution goes over every pair in turn, as a sweep of value
iteration does. The *on-policy* distribution simulates episodes from the start state with the model, ε-greedy on the
current values, and updates each pair as it is met: it spends its updates where the policy goes. Figure 8.8 compares the
two on random tasks of 1000 states, each pair leading to one of `b` next states (its *branches*), with a 0.1 chance of
ending at every step: on-policy updates find a good policy for the start state much sooner, all the more so with few
branches, and the uniform updates catch up later, and pass them.

## Real-time dynamic programming

RTDP (Section 8.7) is the on-policy distribution for value iteration. Each *trial* starts in a start state and follows the
greedy policy on the values, drawing the outcome of every move from the model; at every state it reaches it makes the
value iteration update,

```math
V(s) \leftarrow \max_a \sum_{s', r} p(s', r \mid s, a) \left( r + \gamma \, V(s') \right)
```

and then takes the best of those actions, ties broken at random. The other states are never updated. If the values start
*optimistic* (no lower than the optimal ones, and 0 at the goals, as 0 is everywhere when every step costs -1), every step
costs something, and some policy reaches a goal from every start state for sure, the values of the states that matter
converge to the optimal ones, and the greedy policy becomes optimal from every start
state, without ever sweeping the whole state space. An update keeps an optimistic value optimistic, so RTDP's values never
go below the optimal ones.

## Heuristic search

Heuristic search (Section 8.9) decides in the state the agent is in, by looking `d` moves ahead through every outcome of the
model, and backing the values up from the bottom:

```math
V_d(s) = \max_a \sum_{s', r} p(s', r \mid s, a) \left( r + \gamma \, V_{d-1}(s') \right) \qquad V_0(s) = V(s)
```

The values `V` at the bottom are the *heuristic*. The action taken is the best one at depth `d`: with `d = 1` it is the
greedy action on `V`, and the deeper the search, the less it depends on `V` being right, and the more it costs, about
`(actions * outcomes)^d` backups for every move. `V_d` is exactly what `d` sweeps of value iteration (the two-array version)
would give from `V`, but worked out for one state only: the computation goes where the agent is.

## Rollout algorithms

A rollout algorithm (Section 8.10) estimates each action `a` of the state the agent is in by simulating episodes that start
with `a`, and then follow a *rollout policy* `π` until they end: their average return is a Monte Carlo estimate of
`q_π(s, a)`. Taking the best of them is one step of policy improvement on `π`, made only for this state, when the agent is
there. So, given enough rollouts, the agent plays at least as well as `π`, and often much better, without learning
anything: the estimates are thrown away after each move.

The *given enough rollouts* matters: with few of them, the estimates are noisy, and where `π` is already good the noise can
pick a worse action. On the racetrack below, rolling out a good policy 20 times for each action played anywhere from a
little better to a little worse than the policy itself, depending on the seed; rolling out a poor one was always much
better.

## In relearn

```cpp
using model_type = relearn::finite_mdp<state,action>;   // the model, as for dynamic_programming
model_type model = game.model();
std::mt19937 prng(0);

// trajectory sampling: γ, the distribution and ε; the values go into the policy given
relearn::trajectory_sampling<state,action> sampling(1.0, relearn::update_distribution::on_policy, 0.1);
sampling.plan(model, q, start, 1000, prng);              // 1000 expected updates

// real-time dynamic programming: γ; one trial from a start state
relearn::real_time_dynamic_programming<state,action> rtdp(1.0);
rtdp.trial(model, start, prng);
double v = rtdp.value(s);

// heuristic search: γ and the depth, with a heuristic V
relearn::heuristic_search<state,action> search(1.0, 3);
search.values = rough.values;
action a = search.choose(model, s);

// rollouts: γ, the rollouts of each action, and the rollout policy (at random if not set)
relearn::rollout_planner<state,action> rollouts(1.0, 20, rollout_policy);
action b = rollouts.choose(model, s, prng);
```

- A terminal state is a state with no action, as for `dynamic_programming`, and it is worth 0.
- `trajectory_sampling` keeps its values in the policy given, as the learners do; the uniform updates carry on from where the
  last call stopped, and so does the on-policy trajectory. `real_time_dynamic_programming` and `heuristic_search` keep theirs
  in `values`, a map from each state to its value, as `dynamic_programming` does, a state not in it being worth 0. So the
  values of a few sweeps of `dynamic_programming` can be given to `heuristic_search` as its heuristic, as the example does.
- `real_time_dynamic_programming::updates()` counts its updates, one at every step of every trial, and `values.size()` is the
  number of states it ever updated.
- Values that are equal in exact arithmetic can differ in their last digits after rounding, so the actions within `ties`
  (1e-9) of the best are taken as equally good, as `dynamic_programming` does: RTDP and the on-policy trajectories break such
  ties at random, and heuristic search takes the first of them in `A(s)`. Without this, a different compiler's rounding
  picks a different action, and the racetrack below drives quite differently.
- `rollout_planner::rollout_policy` is given the state, its legal actions and the generator, and returns an action; the
  episodes are drawn from the model (`sample_outcome`), and cut after `max_steps` steps.

## Example: trajectory sampling (Figure 8.8)

`examples/trajectory_sampling.cpp` plans on the book's random tasks (see `examples/games/branching_task.hpp`): 1000 states,
2 actions, `b` branches for each pair, rewards drawn from a normal distribution of mean 0 and variance 1. After each number
of expected updates, the greedy policy on the values is evaluated exactly, from the start state, and averaged over 30 tasks:

```
    updates:               250    1000    2000    5000   10000   20000
    b =  1, uniform       1.101   3.147   5.667   6.978   7.400   7.806
    b =  1, on-policy     4.491   5.658   6.075   6.496   7.018   7.236
    b =  3, uniform       0.667   1.687   3.175   3.395   3.418   3.427
    b =  3, on-policy     1.248   1.992   2.401   2.738   2.872   3.018
    b = 10, uniform       0.363   0.966   1.758   1.790   1.794   1.795
    b = 10, on-policy     0.511   0.967   1.221   1.406   1.484   1.578
```

As in the book, the on-policy updates are ahead at first (after 250 updates, the uniform ones have updated only an
eighth of the pairs), the more so the fewer the branches, and the uniform ones pass them later: with `b = 3` and 10 once
they have swept every pair (2000 updates), and with `b = 1` only between 2000 and 5000 updates. The program succeeds if after 250 updates on-policy is
ahead for `b = 1` and 3, and after 20000 uniform is ahead for `b = 3` and 10.

## Example: the racetrack (Example 8.6)

`examples/racetrack.cpp` plans on the racetrack of Exercise 5.12 (see `examples/games/racetrack.hpp`): a car on a grid,
whose velocity each action changes by -1, 0 or +1 in each direction, which must drive round a turn from the start line to
the finish line, every step costing -1, with a 0.1 chance at every step that the velocity does not change. Running off the
track puts the car back on the start line, at rest. The book does not give its track as text, so the track here is drawn
after it; it has 7575 states.

First Example 8.6, value iteration against RTDP. Value iteration sweeps every state until no value changes by `10⁻⁴`; RTDP
stops (checked every 100 trials) once its greedy policy, evaluated exactly, is within 0.05 steps of the optimal one:

```
    the optimal policy: 11.553 steps from the start line, on average
    value iteration: 16 sweeps, 121200 updates, of every state
    RTDP: 3460 trials, 50733 updates, of 2515 states (33.2%), until its greedy policy is within 0.05 steps of the optimal one (10 runs)
```

RTDP needs 42% of value iteration's updates, and never touches two thirds of the states: the cells far from any good line,
at speeds no good driver reaches there. (In the book's Example 8.6, RTDP needed about half the updates.)

Then planning from rough values, those of a few sweeps of value iteration from 0, all of them policies evaluated exactly
from the start line (steps to the finish):

```
    sweeps     greedy   search 2   search 3   rollouts
         3      never      60.65      25.03      33.20
         8      14.75      11.97      11.63      13.51
    optimal     11.55
```

After 3 sweeps, the greedy policy never finishes: the values only know the last 3 steps before the line, and from some
of the states it reaches the car goes round for ever ("never" is that: the example checks that every state a policy reaches
can still get to the finish line, before evaluating it). Searching deeper finds the way (25 steps 3 moves deep, which is
exactly what the greedy policy on 5 sweeps would drive), and so do rollouts of that greedy policy: each rollout is cut
after 200 steps, so one that goes round for ever scores -200, and the actions that lead to the finish stand out. After 8 sweeps, searching 3 moves deep is within 0.1 step of
the optimal policy.

The program succeeds if RTDP reaches the optimal policy with fewer updates than value iteration, updating fewer than half
the states, if from 8 sweeps searching 3 moves deep beats the greedy policy, and if from 3 sweeps the rollouts do.

__note__: these numbers are GCC's. With Clang's standard library (libc++), RTDP took 2930 trials and 44592 updates, and
Figure 8.8's tasks are drawn differently, so their values differ a little, in the same order; the rest is the same.

## Tests

`tests/planning_test.cpp` checks:

- by hand, on small models: expected updates in the uniform order and along on-policy trajectories (greedy with `ε = 0`
  and exploring with `ε = 1`, ties broken at random, from the start state given, and carried on from one call to the
  next); RTDP's trials, its
  updates and greedy actions, its ties broken at random, and with `γ = 0.9` the optimal values of a corridor; heuristic
  search 1 and 2 moves deep choosing differently, a terminal state worth 0, the first of tied actions; and rollouts,
  discounted, cut at `max_steps`, with a given rollout policy or at random; and outcomes drawn by their probabilities
- against their definitions: uniform expected updates on FrozenLake are exactly sweeps of value iteration on the action
  values, written out in the test, and converge to `q*`; heuristic search `d` moves deep (1 to 4) from a random heuristic
  is exactly `d` sweeps of value iteration, the two-array version, in every state; and the rollouts' estimates are `q` of
  the rollout policy, worked out by policy evaluation, to within 0.032
- convergence: on-policy updates learn `q*` of FrozenLake's start state; on the racetrack, after 5000 trials RTDP's
  values are never below the optimal ones, it has updated fewer than half the states (2512 to 2523 of 7575), and its
  greedy policy is within 0.2 steps of the optimal one (0.013 to 0.055 over 10 seeds); Figure 8.8, with 10 tasks: after 250
  updates with `b = 1` on-policy was ahead by 1.48 or more, and after 20000 with `b = 10` uniform by 0.19 to 0.23
- planning from rough values on the racetrack: from 8 sweeps, the greedy policy takes 14.75 steps and a search 3 moves deep
  11.63, exactly; from 3 sweeps, the rollouts of the greedy policy (which never finishes) take 25 to 33 steps, over 10 seeds
- every one of 29 mutations (no `γ`; a terminal state not worth 0; the uniform updates skipping actions; the on-policy
  trajectories always exploring, or with `ε` the wrong way round, or starting again at every update, or at every call, or
  from another state; a branch of the random tasks leading to the end; RTDP's value from the first action, or its ties
  always to the first; a search one move too deep, or with leaves worth 0; rollouts without their first reward, or
  ignoring the rollout policy; the racetrack's path checked only at its end, or its finish line only there; and so on)
  makes at least one of these tests fail

and `tests/games_test.cpp` checks the racetrack's rules (moving, crashing, the finish line crossed before the wall, a path
clipping the corner of a wall, the legal actions, the noise) and its model against the game, every move of every state, and
the random tasks of Section 8.6. Each threshold was set from 10 seeds, with GCC's standard library and with Clang's (libc++).

## References

1. Sutton, R.S. and Barto, A.G., 2018. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 2nd edition. Cambridge, MA: MIT Press.
   Section 4.4 (value iteration), Section 8.6 (trajectory sampling, Figure 8.8), Section 8.7 (real-time dynamic programming,
   Example 8.6), Section 8.9 (heuristic search), Section 8.10 (rollout algorithms), Exercise 5.12 (the racetrack).
2. Barto, A.G., Bradtke, S.J. and Singh, S.P., 1995. Learning to act using real-time dynamic programming. *Artificial
   Intelligence*, 72(1-2), 81-138.
