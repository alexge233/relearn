#ifndef RELEARN_MCTS_HPP
#define RELEARN_MCTS_HPP
/**
 * Copyright 2015 Alexander Giokas
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * #http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file relearn/mcts.hpp
 * @brief Monte Carlo Tree Search, with UCT as the tree policy (`mcts`)
 */
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace relearn
{
/**
 * @struct mcts This is Monte Carlo Tree Search (Sutton & Barto, Section 8.11),
 *         with UCT (Kocsis and Szepesvári, 2006) as the tree policy
 * @brief plans one move at a time: it plays many games in a model of the world,
 *        and grows a tree of the positions reached, spending more games on the moves that look best
 * @param exploration is c, the exploration constant of UCT
 * @param gamma is the discount rate
 * @param iterations is the number of iterations (simulated games) of each search
 * @param rollout_limit is the most steps a rollout may take
 *
 * MCTS learns nothing that lasts: each time it has to move, it builds a new tree from the
 * current state, picks a move, and throws the tree away. Each iteration has four phases:
 *
 * 1. selection: from the root, whilst every action of the node has been tried, go down to the
 *    child of highest UCT value (the tree policy)
 *
 *    Q(child) + c * sqrt(ln N(node) / N(child))
 *
 *    where N counts the visits of a node and Q is the average return after it
 * 2. expansion: add one child to the node reached, for its first action not tried yet
 * 3. simulation: from the new child, play at random to the end (the rollout, the default policy)
 * 4. backup: walk back up to the root, adding the return to every node on the way
 *
 * The return is worked out backwards, as Monte Carlo does, G_t = r_t + γ * G_{t+1}, except that
 * each value is from the point of view of the player who moved: when the player to move changes,
 * the sign of G_{t+1} changes too (negamax, for two players who play against each other)
 *
 * G_t = r_t + γ * G_{t+1}     if the same player moves at t and t+1
 * G_t = r_t - γ * G_{t+1}     if the other player moves at t+1
 *
 * so Q(child) is always the value of the move for the player who made it, and the
 * player choosing at a node simply takes the highest. A one-player task never changes sign.
 * The change of sign assumes two players who play against each other: what one wins, the other loses.
 *
 * Once the iterations are done, the move played is the most visited child of the root
 * (one of the choices Sutton & Barto mention), rather than the one of highest value: a value
 * resting on few games can be high by luck. Between children visited as often, the one of
 * higher value is played.
 *
 * The model is what MCTS plans with, and `model_type` must provide:
 *
 *     using state_type  = ...;                                 // copyable
 *     using action_type = ...;                                 // copyable
 *     std::vector<action_type> actions(const state_type & s);  // the legal actions in s
 *     result step(const state_type & s, const action_type & a);
 *     unsigned int player(const state_type & s);               // the player to move in s
 *
 * where `result` has the members `next` (the state reached), `reward` (the reward received by
 * the player who took a in s) and `terminated` (true if the game ended), e.g. `games::step_result`.
 * For a one-player task, `player` simply returns 0. A state with no legal actions counts as the end.
 */
template <class model_type,
          typename value_type = double>
struct mcts
{
    using state_type  = typename model_type::state_type;
    using action_type = typename model_type::action_type;

    /**
     * @struct node
     * @brief one node of the search tree: a state reached, and what the iterations found after it
     */
    struct node
    {
        /// the state of the node
        state_type state;
        /// the player to move in the state
        unsigned int player;
        /// r: the reward received by the player who moved into this node (0 for the root)
        value_type reward;
        /// true if the game ended on the way into this node
        bool terminal;
        /// the legal actions in the state (none if terminal), in the order they are tried
        std::vector<action_type> actions;
        /// the children so far: `children[i]` is the index, in the tree, of the node `actions[i]` leads to
        std::vector<std::size_t> children;
        /// N: the number of iterations that went through this node
        std::size_t visits;
        /// the sum of their returns, from the point of view of the player who moved into this node
        value_type total;

        /// @return Q, the average return after this node (0 before any visit)
        value_type value() const;
    };

    /**
     * c: the exploration constant of UCT, √2 by default (UCB1's, for returns between 0 and 1).
     * Scale it by the range of the returns, e.g. 2√2 for returns between -1 and 1
     */
    value_type exploration = std::sqrt(value_type(2));
    /// discount rate - 1 by default, as games usually are not discounted
    value_type gamma = 1;
    /// the number of iterations of each search, i.e. of each move
    unsigned int iterations = 1000;
    /// the most steps a rollout may take: a rollout cut short counts only the rewards it received
    std::size_t rollout_limit = 1000;
    /// the model of the world: MCTS calls its `actions`, `step` and `player`
    model_type model;
    /// draws the moves of the rollouts - seed it to repeat a search
    std::mt19937 prng;

    /**
     * @brief ctor with the @param initial_model to plan with,
     *        the @param initial_iterations of each search (1000 by default),
     *        the @param initial_exploration constant c (√2 by default),
     *        and the @param prng_seed of the generator that draws the rollouts
     */
    mcts(model_type initial_model,
         unsigned int initial_iterations = 1000,
         value_type initial_exploration = std::sqrt(value_type(2)),
         unsigned int prng_seed = 0);

    /**
     * @brief search from @param root, and @return the action to take there
     * @note it is `search(root)` then `best_action()`
     */
    action_type operator()(const state_type & root);

    /// @brief build a new tree from @param root, with `iterations` iterations
    void search(const state_type & root);

    /**
     * @return the most visited action of the root; between actions visited as often, the one of
     *         higher value (and then the first of them)
     * @warning call `search` first
     */
    action_type best_action() const;

    /// @return the tree of the last search: `tree()[0]` is the root
    const std::vector<node> & tree() const;

private:
    // one iteration: selection, expansion, simulation and backup
    void iterate();
    // a node for state s, reached with reward r, `terminal` if the game ended
    node make_node(const state_type & s,
                   value_type r,
                   bool terminal);
    // selection: the child of `parent` of highest UCT value
    std::size_t select(std::size_t parent) const;
    // expansion: add the child of `parent` for its first action not tried yet
    std::size_t expand(std::size_t parent);
    // simulation: play at random from the state of `leaf`, and return its return
    value_type rollout(std::size_t leaf);
    // the tree of the last search: tree_[0] is the root
    std::vector<node> tree_;
};

/********************************************************************************
 ***********               Template Implementations
 ********************************************************************************/

template <class model_type,
          typename value_type>
value_type mcts<model_type,value_type>::node::value() const
{
    // Q = W / N, the average of the returns
    if (visits == 0) {
        return 0;
    }
    return total / static_cast<value_type>(visits);
}

template <class model_type,
          typename value_type>
mcts<model_type,value_type>::mcts(model_type initial_model,
                                  unsigned int initial_iterations,
                                  value_type initial_exploration,
                                  unsigned int prng_seed)
: exploration(initial_exploration),
  iterations(initial_iterations),
  model(initial_model),
  prng(prng_seed)
{}

template <class model_type,
          typename value_type>
typename mcts<model_type,value_type>::action_type
            mcts<model_type,value_type>::operator()(const state_type & root)
{
    search(root);
    return best_action();
}

template <class model_type,
          typename value_type>
void mcts<model_type,value_type>::search(const state_type & root)
{
    // a new tree, with only the root in it
    tree_.clear();
    tree_.push_back(make_node(root, 0, false));
    assert(!tree_[0].terminal && "mcts needs a root with legal actions");
    for (unsigned int i = 0; i < iterations; i++) {
        iterate();
    }
}

template <class model_type,
          typename value_type>
typename mcts<model_type,value_type>::action_type
            mcts<model_type,value_type>::best_action() const
{
    assert(!tree_.empty() && !tree_[0].children.empty() && "call mcts::search first");
    // the most visited child: its value rests on the most games, and UCT
    // spends most of its games on the move it believes is best
    const node & root = tree_[0];
    std::size_t best = 0;
    for (std::size_t i = 1; i < root.children.size(); i++) {
        const node & c = tree_[root.children[i]];
        const node & b = tree_[root.children[best]];
        // between children visited as often, the one of higher value
        if (c.visits > b.visits || (c.visits == b.visits && c.value() > b.value())) {
            best = i;
        }
    }
    return root.actions[best];
}

template <class model_type,
          typename value_type>
const std::vector<typename mcts<model_type,value_type>::node> &
            mcts<model_type,value_type>::tree() const
{
    return tree_;
}

template <class model_type,
          typename value_type>
void mcts<model_type,value_type>::iterate()
{
    // 1. selection: go down the tree whilst every action of the node has been tried
    std::vector<std::size_t> path(1, 0);
    while (!tree_[path.back()].terminal &&
           tree_[path.back()].children.size() == tree_[path.back()].actions.size()) {
        path.push_back(select(path.back()));
    }
    // 2. expansion: unless the game is over, add a child for an action not tried yet
    if (!tree_[path.back()].terminal) {
        path.push_back(expand(path.back()));
    }
    // 3. simulation: the return after the new node, from the point of view of the player to move there
    value_type G = tree_[path.back()].terminal ? 0 : rollout(path.back());
    // 4. backup: walk back to the root, from the point of view of the player who moved into each node:
    //    G_t = r_t + γ G_{t+1} if the same player moves next, r_t - γ G_{t+1} if the other one does
    for (std::size_t i = path.size() - 1; i > 0; i--) {
        node & child = tree_[path[i]];
        const node & parent = tree_[path[i - 1]];
        G = child.reward + gamma * (child.player == parent.player ? G : -G);
        child.visits++;
        child.total += G;
    }
    // the root counts its visits too, for the ln N of UCT
    tree_[0].visits++;
}

template <class model_type,
          typename value_type>
typename mcts<model_type,value_type>::node
            mcts<model_type,value_type>::make_node(const state_type & s,
                                                   value_type r,
                                                   bool terminal)
{
    node n = {s, model.player(s), r, terminal, {}, {}, 0, 0};
    // the legal actions, unless the game is over; a state without any is the end too
    if (!terminal) {
        n.actions = model.actions(s);
        n.terminal = n.actions.empty();
    }
    return n;
}

template <class model_type,
          typename value_type>
std::size_t mcts<model_type,value_type>::select(std::size_t parent) const
{
    // UCT: the child of highest Q(child) + c * sqrt(ln N(parent) / N(child)),
    // where Q(child) is from the point of view of the player to move in the parent
    const node & p = tree_[parent];
    const value_type log_n = std::log(static_cast<value_type>(p.visits));
    std::size_t best = p.children.front();
    value_type best_score = 0;
    for (std::size_t i = 0; i < p.children.size(); i++) {
        const node & c = tree_[p.children[i]];
        value_type score = c.value()
                         + exploration * std::sqrt(log_n / static_cast<value_type>(c.visits));
        // the first child of highest value wins ties
        if (i == 0 || score > best_score) {
            best = p.children[i];
            best_score = score;
        }
    }
    return best;
}

template <class model_type,
          typename value_type>
std::size_t mcts<model_type,value_type>::expand(std::size_t parent)
{
    // the first action of the parent not tried yet, taken in the model
    const action_type a = tree_[parent].actions[tree_[parent].children.size()];
    auto out = model.step(tree_[parent].state, a);
    // the child joins the tree, as the parent's node for that action
    tree_.push_back(make_node(out.next, static_cast<value_type>(out.reward), out.terminated));
    tree_[parent].children.push_back(tree_.size() - 1);
    return tree_.size() - 1;
}

template <class model_type,
          typename value_type>
value_type mcts<model_type,value_type>::rollout(std::size_t leaf)
{
    // play at random from the leaf, remembering each reward and the player who received it
    std::vector<value_type> rewards;
    std::vector<unsigned int> movers;
    state_type s = tree_[leaf].state;
    unsigned int player = tree_[leaf].player;
    while (rewards.size() < rollout_limit) {
        auto legal = model.actions(s);
        if (legal.empty()) {
            break;
        }
        // the default policy: any legal action, at random
        std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
        auto out = model.step(s, legal[pick(prng)]);
        rewards.push_back(static_cast<value_type>(out.reward));
        movers.push_back(player);
        if (out.terminated) {
            break;
        }
        s = out.next;
        player = model.player(s);
    }
    // the return, backwards: G_t = r_t ± γ G_{t+1}, the sign changing with the player.
    // After the last step G is 0, so its sign does not matter
    value_type G = 0;
    for (std::size_t t = rewards.size(); t-- > 0; ) {
        bool same = t + 1 == rewards.size() || movers[t + 1] == movers[t];
        G = rewards[t] + gamma * (same ? G : -G);
    }
    // the return of the player to move at the leaf
    return G;
}
} // end of namespace
#endif
