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
 * policies have a negative value.
 *
 * For this type of approach to work, in between episodes, the agent must
 * be rewarded and re-trained.
 */
template <typename S, 
          typename A>
std::deque<relearn::link<S,A>> explore(const world & w,
                                       std::mt19937 & gen,
                                       relearn::policy<S,A> & policy_map,
                                       grid start)
{
    // explore until we discover a negative or positive reward!
    bool stop = false;
    // the markov_chain/episode we populate
    std::deque<relearn::link<S,A>> episode;

    // S_t (state now) is initially the root state
    std::cout << "starting exploration from: " << start.x 
              << "," << start.y << std::endl;
    grid curr = start;
    auto state_now = state(curr.R, curr);

    // explore while if no policy is found
    // and keep populating the episode with states and actions
    // if there exists a policy, then stay on it!
    while (!stop) 
    {
        auto pair   = policy_map.best(state_now);
        if (pair.first && pair.second > 0) {
            switch (pair.first->trait().dir) {
                case 0 : curr.y--;
                         break;
                case 1 : curr.x++;
                         break;
                case 2 : curr.y++;
                         break;
                case 3 : curr.x--;
                         break;
            }
            auto it = w.blocks.find(curr);
            if (it != w.blocks.end()) {
                curr = *it;
                auto state_next = S(curr.R, curr);
                state_now = state_next;
                std::cout << "coord: " << curr.x << "," 
                          << curr.y << " = " << curr.R << std::endl;

                if (curr.R == -1.0 || curr.R == 1.0) {
                    break;
                }
            }
        }
        else {
            // randomly decide on next grid - we map numbers to a direction
            auto result = rand_direction()(gen, w, curr);
            // find the reward at the current coordinates
            curr = result.second;
            // create the action using direction as trait
            auto action_now = A(result.first);
            // add the state to the episode
            episode.emplace_back(relearn::link<S,A>{state_now, action_now});
            // update current state to next state
            state_now = state(curr.R, curr);
            std::cout << "coord: " << curr.x << "," 
                      << curr.y << " = " << curr.R << std::endl;

            if (curr.R == -1 || curr.R == 1) {
                break;
            }
        }
    }
    // Add the terminal state last
    // note: we don't have an empty action - we could use pointers instead...
    auto action_empty = action(direction({100}));
    episode.emplace_back(relearn::link<state,action>{state_now, action_empty});
    return episode;
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
        // use Q-learning algorithm to update the episode's policies
        auto learner = relearn::q_learning<state,action>{0.9, 0.9};
        for (int k = 0; k < 10; k++) {
            learner(episode, policies);
        }
        stop = (episode.back().state.reward() == 1 ? true : false);
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
