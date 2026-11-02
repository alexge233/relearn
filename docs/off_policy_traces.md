# Off-policy traces: off-policy TD(λ), Expected Sarsa(λ) and Tree-Backup(λ)

Eligibility traces (have a look at [eligibility_traces.md](eligibility_traces.md)) share every TD error with the
steps that came before it. Off-policy, the steps were taken by a *behaviour* policy `b`, and the values to learn are
those of another, the *target* policy `π`: a greedy one, say, learnt from an agent that explores. Each step then has to
be weighted by how likely `π` was to take it. Sutton and Barto [1] give two ways of doing it with traces, in Sections
12.9 and 12.10, and relearn has both:

- **importance sampling** (Section 12.9): each step weighted by its ratio `ρ_t = π(a_t | s_t) / b(a_t | s_t)`, one step at
  a time (*per-decision*), with a *control variate* that keeps the value of a step `π` would rarely take, instead of
  pulling it to 0. `off_policy_td_lambda` learns state values this way, and `off_policy_expected_sarsa_lambda` action values.
- **tree backup** (Section 12.10): each step weighted by `π(a_t | s_t)` alone, the way n-step tree backup does
  ([n_step.md](n_step.md)), without `b` at all: `tree_backup_lambda`.

Watkins's Q(λ) ([eligibility_traces.md](eligibility_traces.md)) is a third way, for a greedy `π` only: its traces are
cut after every move that is not greedy.

## Importance sampling, a step at a time

For state values, the λ-return of Section 12.9 weights what follows each state by its ratio, and keeps the value of the
state for the rest of the weight:

```math
G^{\lambda s}_t = \rho_t \left( r_t + \gamma_{t+1} \left( (1 - \lambda_{t+1}) \, V(s_{t+1}) + \lambda_{t+1} \, G^{\lambda s}_{t+1} \right) \right) + (1 - \rho_t) \, V(s_t)
```

where, as always, `r_t` is the reward received after leaving `s_t` (have a look at the
[reward convention](q_learning.md#reward-convention)), and `γ_t = γ(s_t)` and `λ_t = λ(s_t, a_t)` may vary from state to
state (Section 12.8). The `(1 - ρ_t) V(s_t)` term is the control variate: a step `π` would never take (`ρ_t = 0`) leaves
`V(s_t)` where it is. Its backward view, *off-policy TD(λ)*, is TD(λ) with the ratio in the trace:

```math
\delta_t = r_t + \gamma_{t+1} \, V(s_{t+1}) - V(s_t) \qquad z(s) \leftarrow \rho_t \left( \gamma_t \lambda_t \, z(s) + [s = s_t] \right) \qquad V(s) \leftarrow V(s) + \alpha \, \delta_t \, z(s)
```

For action values, the TD error is Expected SARSA's, with `V̄(s) = Σ_a π(a | s) Q(s, a)`, and the ratio fades the traces
from the next step on, since `Q(s_t, a_t)` is already the value of the action taken:

```math
\delta_t = r_t + \gamma_{t+1} \, \bar V(s_{t+1}) - Q(s_t, a_t) \qquad z(s, a) \leftarrow \gamma_t \lambda_t \rho_t \, z(s, a) + [s = s_t, a = a_t] \qquad Q(s, a) \leftarrow Q(s, a) + \alpha \, \delta_t \, z(s, a)
```

whose λ-return is

```math
G^{\lambda a}_t = r_t + \gamma_{t+1} \left( (1 - \lambda_{t+1}) \, \bar V(s_{t+1}) + \lambda_{t+1} \left( \rho_{t+1} \, G^{\lambda a}_{t+1} + \bar V(s_{t+1}) - \rho_{t+1} \, Q(s_{t+1}, a_{t+1}) \right) \right)
```

Offline, if the values do not change during the episode, the TD errors add up to exactly the λ-return's error, and the
two make the very same changes; online, nearly. With `λ = 0` both are one-step methods (Expected SARSA, for the action
values), and with `π = b` every ratio is 1 and off-policy TD(λ) is TD(λ).

**The traces can grow.** A ratio is above 1 whenever `π` takes an action more often than `b`: with a greedy `π` and an
agent that picks one of 4 moves at random, `ρ = 4` at every greedy move, and the traces grow by `γλρ` at every step.
With `γ = 0.95` and `λ = 0.9` that is 3.4, and in about half the runs the values blow up (see the
[example](#example-frozenlake-from-random-play) below); with `λ = 0.25` it is 0.95, and they don't. This is the variance of importance sampling (have a look at
[monte_carlo.md](monte_carlo.md)), a step at a time.

## Tree-Backup(λ)

Tree backup weights each step by `π` alone. After each reward, it takes the values of the actions `π` might have taken
instead, and carries on with the return of the action taken, each weighted by its probability:

```math
G^{\lambda a}_t = r_t + \gamma_{t+1} \left( (1 - \lambda_{t+1}) \, \bar V(s_{t+1}) + \lambda_{t+1} \left( \sum_{a \neq a_{t+1}} \pi(a \mid s_{t+1}) \, Q(s_{t+1}, a) + \pi(a_{t+1} \mid s_{t+1}) \, G^{\lambda a}_{t+1} \right) \right)
```

Its backward view is the one above, with `π(a_t | s_t)` in place of `ρ_t`: the traces fade by `γ_t λ_t π(a_t | s_t)`.
Probabilities are at most 1, so the traces only ever shrink, and whatever `b` was, nothing blows up. The price is that
they shrink even on-policy, whenever `π` is not close to deterministic, so tree backup looks less far ahead than its λ
says. With a greedy `π` its traces are cut after a move that is not greedy, much as Watkins's Q(λ)'s are.

When `b` is uniform, `1/k` for each of `k` actions, the two are one: `ρ_t = k π(a_t | s_t)`, so Tree-Backup(λ) is
importance sampling with `λ / k`, exactly.

## In relearn

```cpp
// off-policy TD(λ): the state values of π, learnt from episodes played by b
auto pi = [](const state & s, const action & a) { return a.trait() == 0 ? 0.85 : 0.05; };  // π(a | s)
auto b  = [](const state & s, const action & a) { return 0.25; };  // b(a | s)
relearn::off_policy_td_lambda<state,action> td(0.02, 0.95, 0.5, pi, b);   // α, γ, λ, π and b
auto r = games::play(game, agent, 1000);
td(r.episode, r.terminated);
double v = td.value(s);
// off-policy Expected Sarsa(λ): π as a distribution (greedy if not set), and b
relearn::off_policy_expected_sarsa_lambda<state,action> sampled(0.1, 0.95, 0.25, nullptr, b);
sampled(r.episode, memory, r.terminated);
// Tree-Backup(λ): π alone (greedy if not set)
relearn::tree_backup_lambda<state,action> tree(0.1, 0.95, 0.9);
tree(r.episode, memory, r.terminated);
```

- **Each call learns one whole episode**, as `sarsa_lambda` does: the traces start at 0, and every step is learnt in
  order. So play the episode with `games::play`, then give it to the learner. An episode cut short
  (`terminated = false`) ends with the state reached and the action chosen next: it is not learnt as a step, and only
  gives the state the step before it bootstraps from.
- `off_policy_td_lambda` keeps its values itself, `value(s)`, since an off-policy prediction task has several actions
  per state, and relearn's policies hold the values of state/action pairs. It needs `π` and `b`, as functions of a state
  and an action, as `off_policy_monte_carlo_prediction` does.
- `off_policy_expected_sarsa_lambda` and `tree_backup_lambda` learn into the policy you give them. `probabilities` is `π`,
  as in `expected_sarsa`: every action `π` may take in a state, with its probability. If it is not set, `π` is greedy on
  the values, the actions of highest value sharing the probability, and `V̄(s)` is the highest value of `s`; an action
  whose value is not stored yet has `π = 0`, so early on the traces are cut at every state not seen before (Watkins's
  Q(λ) counts such a move as greedy).
  `off_policy_expected_sarsa_lambda` needs `b` too, `behaviour`, which must be above 0 for every action it took.
- `π` is read with the values as they are at each step, and `b` for every step of the episode once, before anything is
  learnt, since the episode was played with it.
- `learner.offline = true` works out every `δ` (and `π`) with the values from before the episode: they then make
  exactly the changes of their λ-returns.
- `lambda_of(s, a)` and `gamma_of(s)` give a λ and a γ for every state, as for the other learners with traces.
- `learner.traces` keeps the traces of the last episode learnt (for `off_policy_td_lambda`, those of `(s, action 0)`).

## Example: FrozenLake from random play

`examples/frozen_lake_off_policy.cpp` learns slippery 4x4 FrozenLake (see `examples/games/frozen_lake.hpp`, `γ = 0.95`)
from an agent that only ever moves at random, `b = 1/4` for every move. The true values come from value iteration on the
game's model, `frozen_lake::model()`. First the optimal values `q*`, with a greedy `π` (the RMS error of `Q`, 20 runs):

```
    episodes:                                 100       1000      10000
    Expected SARSA (λ = 0)                 0.2814     0.2062     0.0487
    Watkins's Q(λ), λ = 0.9                0.2714     0.1837     0.0487
    Tree-Backup(λ), λ = 0.5                0.2809     0.1990     0.0470
    Tree-Backup(λ), λ = 0.9                0.2804     0.1912     0.0490
    Tree-Backup(λ), λ = 1                  0.2803     0.1888     0.0491
    importance sampling, λ = 0.25          0.2803     0.1888     0.0491
    importance sampling, λ = 0.9           0.2875     0.5250    59.4693
```

The traces help early on (after 1000 episodes), and Watkins's Q(λ) and Tree-Backup(λ) end where Expected SARSA does. Tree-Backup(1) and
importance sampling with `λ = 0.25` learn the very same values (their largest difference is 0), as they should with a
uniform `b`. And importance sampling with `λ = 0.9` blows up in about half the runs, taking the mean error with it: its
traces grow by 3.4 at every greedy move.

Then the state values of a policy that takes the optimal move with 0.85 and each other one with 0.05, with off-policy
TD(λ) (`α = 0.02`, the RMS error of `V`, 20 runs):

```
    episodes:                                 100       1000      10000
    off-policy TD(λ), λ = 0.0              0.2587     0.2005     0.0311
    off-policy TD(λ), λ = 0.5              0.2555     0.1604     0.0417
    off-policy TD(λ), λ = 0.9              0.2594    69.7717    56.4070
```

With `λ = 0.9`, `γλρ = 0.95 * 0.9 * 3.4 = 2.9` at every move `π` prefers, and in about half the runs the values blow up too.

The program succeeds if Tree-Backup(0.9) ends within 0.07 of `q*`, if Tree-Backup(1) and importance sampling with
`λ = 0.25` learn the same values to within `10⁻¹²`, if importance sampling with `λ = 0.9` ends further than 1 away on
average, and if off-policy TD(λ) ends within 0.08 of the true values with `λ = 0` and 0.1 with 0.5, and further than 1
with 0.9. In 9 sets of 20 runs each, 11 to 14 runs of importance sampling with `λ = 0.9` and 8 to 11 of off-policy TD(λ)
with `λ = 0.9` ended above 1, and the others did not blow up at all.

__note__: these numbers are GCC's. With Clang's standard library (libc++), whose random numbers differ, they are a little
different, in the same order.

## Tests

`tests/off_policy_traces_test.cpp` checks:

- values and traces worked out by hand, for all three on the same episode, and a step `π` would never take (`ρ = 0`)
  cutting the traces of off-policy TD(λ), and leaving the value of its state where it was; a greedy `π` cutting the
  traces of Tree-Backup(λ) after a move that is not greedy, and offline, a greedy `π` read from the values from before
  the episode; and the traces of importance sampling growing above 1
- against the book's definitions, written out in the test: offline, each of the three makes exactly the changes of its
  λ-return of Section 12.9 or 12.10, on 300 trials of 5 random episodes learnt one after the other, with random α, γ, λ,
  `π` and `b` (`π` 0 or 1 in some states, `γ = 1` and `λ = 1` too, a third of the episodes cut short, and every other
  trial with a random `λ(s, a)` and `γ(s)`), to within `10⁻¹²` of the value (relative, when it is above 1, since the
  ratios can make the values large)
- exact equivalences, to within `10⁻¹²`: with `λ = 0` both action-value learners are `expected_sarsa`; with `π = b`
  off-policy TD(λ) is `td_lambda` on the states; with a uniform `b`, Tree-Backup(λ) is importance sampling with `λ b`,
  online and offline; and with `λ = 1`, offline, on episodes that never come back to a state, importance sampling is
  n-step Q(σ) with `σ = 1` and Tree-Backup(λ) n-step tree backup, both to the end of the episode
- that each call is a whole episode, its traces starting at 0, and that an episode cut short does not learn its last link
- control from random play, against exact answers from `tests/solvers.hpp`, in a slippery corridor: with a greedy `π`,
  Tree-Backup(λ) (λ = 0.9) and importance sampling (λ = 0.5) learn `q*` (worst error 0.022 to 0.072 and 0.024 to 0.076)
- prediction from random play on slippery FrozenLake: off-policy TD(λ) learns the values of a policy that takes the
  optimal move with 0.85, against value iteration and policy evaluation on the game's model (mean RMS error 0.022 to
  0.047 with `λ = 0`, 0.037 to 0.063 with `λ = 0.5`); and `tests/games_test.cpp` checks `frozen_lake::model()` against
  the model written from Gymnasium's rules, on both maps, slippery or not
- every one of 28 mutations (no `ρ`, or `ρ` on the old traces only, or without `π`; SARSA's target instead of `V̄`; no `π`
  in tree backup's fading; replacing traces; the traces kept from the last episode; the constant `γλ`; offline ignored;
  an episode cut short learning its last link; and three of the model of FrozenLake, and so on) makes at least one of
  these tests fail

Each threshold of the tests was set from 10 seeds, with GCC's standard library and with Clang's (libc++).

## References

1. Sutton, R.S. and Barto, A.G., 2018. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), 2nd edition. Cambridge, MA: MIT Press.
   Section 7.5 (n-step tree backup), Section 7.6 (n-step Q(σ)), Section 12.8 (a λ and a γ that vary), Section 12.9
   (off-policy traces with control variates), Section 12.10 (Watkins's Q(λ) and Tree-Backup(λ)).
2. Precup, D., Sutton, R.S. and Singh, S., 2000. Eligibility traces for off-policy policy evaluation. In *Proceedings of
   the 17th International Conference on Machine Learning (ICML)*.
