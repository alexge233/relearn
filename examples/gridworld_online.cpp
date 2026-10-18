/**
 * the Gridworld problem as described by Sutton and Barto.
 * This is a toy environment called Gridworld that is often used 
 * as a toy model in the Reinforcement Learning literature. 
 * In this particular case:
 *
 * 1. State space: GridWorld has 10x10 = 100 distinct states. 
 *    The start state is the top left cell. 
 *    The gray cells are walls and cannot be moved to.
 *
 * 2. Actions: The agent can choose from up to 4 actions to move around.
 *
 * 3. Environment Dynamics: GridWorld is deterministic, 
 *    leading to the same new state given each state and action.
 *    Non-deterministic environments nead a probabilistic approach.
 *
 * 4. Rewards: The agent receives +1 reward when it is in 
 *    the center square (the one that shows R 1.0), 
 *    and -1 reward in a the boundary states (R -1.0 is shown for these). 
 *    The state with +1.0 reward is the goal state.
 *
 * This is a deterministic, finite Markov Decision Process (MDP) 
 * and the goal is to find an agent policy that maximizes 
 * the future discounted reward.
 *
 * This version of the Gridworld example uses on-line on-policy decision-making
 * What that means is that as the agent moves, it tries to use the already known
 * policy, unless it has a bad value, or if it is unknown, in which case
 * it takes a random action.
 */
#include "gridworld_header.hpp"
// set shortcuts to state trait and action trait
using state = relearn::state<grid>;
using action = relearn::action<direction>;

/**
 * Exploration technique is `online` meaning the agent will follow
 * policies, if they exist for a particular state. It will revert to a 
 * stochastic (random) approach only if policies don't exist, or if those
 * policies have a negative value, or if they lead into a wall.
 *
 * Every step is recorded in the episode, whether it followed the policy or not,
 * with the reward of the block the agent moved into.
 *
 * An episode cut short by the step limit is returned empty: its last move did
 * not end the game, but `q_learning` treats the last link of an episode as
 * final, so training on it would teach wrong values.
 * For this type of approach to work, in between episodes, the agent must
 * be rewarded and re-trained.
 */
template <typename S, 
          typename A>
std::deque<relearn::link<S,A>> explore(const world & w,
                                       std::mt19937 & gen,
                                       const relearn::policy<S,A> & policy_map,
                                       grid start)
{
    // an episode that wanders for too long without ending is cut short
    const unsigned int max_steps = 1000;
    // the markov_chain/episode we populate
    std::deque<relearn::link<S,A>> episode;

    std::cout << "starting exploration from: " << start.x 
              << "," << start.y << std::endl;
    grid curr = start;

    for (unsigned int step = 0; step < max_steps; step++)
    {
        unsigned int dir = 0;
        const grid * next = nullptr;
        // stay on policy if there is a positive one that does not hit a wall
        auto pair = policy_map.best(S(curr));
        if (pair.first && pair.second > 0) {
            dir  = pair.first->trait().dir;
            next = w.move(curr, dir);
        }
        // otherwise randomly decide on next grid
        grid random_block;
        if (!next) {
            auto result  = rand_direction()(gen, w, curr);
            dir          = result.first.dir;
            random_block = result.second;
            next         = &random_block;
        }
        // record (s_t, a_t) with the reward of the block we moved into
        episode.emplace_back(relearn::link<S,A>{S(next->R, curr), A(direction{dir})});
        curr = *next;
        std::cout << "coord: " << curr.x << "," 
                  << curr.y << " = " << curr.R << std::endl;
        if (terminal(curr)) {
            return episode;
        }
    }
    std::cout << "episode cut short after " << max_steps << " steps: discarded" << std::endl;
    return {};
}

/**
 * Gridworld example main function.
 * The agent will begin with stochastic exploration, 
 * and at every terminal state will update its experience.
 *
 * Then, it learns from that episode, and next time it explores
 * it will take "good" actions and avoid bad ones, 
 * and if no good actions exist, it will move randomly.
 *
 * This form of approach is online on-policy.
 */
int main()
{
    std::mt19937 gen(static_cast<std::size_t>(
        std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count()));

    // create the world and populate it randomly
    world w = populate();
    // hardcoded start grid
    grid start = {1, 8, 0};
    // store policies and episodes - we'll use policies later 
    relearn::policy<state,action> policies;

    bool stop = false;
    // explore the grid and update at the end of the episode
    // next time we explore we can avoid bad actions !
    while (!stop) {
        auto episode = explore<state,action>(w, gen, policies, start);
        if (episode.empty()) {
            continue;
        }
        // use Q-learning algorithm to update the episode's policies
        auto learner = relearn::q_learning<state,action>{0.9, 0.9};
        for (int k = 0; k < 10; k++) {
            learner(episode, policies);
        }
        stop = (!episode.empty() && episode.back().state.reward() == 1);
    }
    //
    // The catch here is that although we explore and stay on policy at the same
    // time, if we do not explore long enough, we may find "a solution" which
    // is not necessarily the optimal or best one!
    // the way to avoid doing this, is to not exit the loop as soon
    // as a solution is found, and in conjunction with this,
    // to also allow random actions, even when an optimal policy exists (this
    // method is known as e-Greedy or explorative-Greedy).
    //
    return 0;
}
