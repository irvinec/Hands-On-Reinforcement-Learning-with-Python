import gym
import numpy as np

NUM_VALUE_ITERATIONS = 10000
STATE_VALUE_TABLE_DELTA_THRESHOLD_FOR_CONVERGENCE = 1e-20
DISCOUNT_FACTOR = 1.0

def main():
    env = gym.make('FrozenLake-v0')
    # Get the value function table for the optimal policy
    optimal_value_table = value_iteration(env)

    # Now that we have the optimal value function in terms of a table (value_table)
    # we need to extract the optimal policy so that we can choose the right action.
    optimal_policy = extract_policy(env, optimal_value_table)

    # Now play the game with the optimal policy
    total_reward = 0
    current_state = env.reset()
    env.render()
    for _ in range(100):
        best_action = optimal_policy[current_state]
        current_state, reward, is_done, _ = env.step(best_action)
        env.render()
        total_reward += reward
        if is_done:
            print('We reached a terminal state!')
            break

    print('The game is finished.')
    print(f'Total reward: {total_reward}')

def value_iteration(env):
    # Do value iteration algorithm
    # Psuedo-code:
    # Initialize V_0(s) to arbitrary values.
    # while V_i(s) not converged:
    #   for s in States:
    #       for a in Actions:
    #           Q(s,a) <- R(s,a) + DISCOUNT_FACTOR*sum(T(s,a,s')*V_i(s'))
    #       V_i+1(s) <- argmax(Q(s,a))

    # Initialize value table to all zeros.
    value_table = np.zeros(env.observation_space.n)

    for iter_index in range(NUM_VALUE_ITERATIONS):
        # Save the old value_table values so we can compare for convergence.
        old_value_table = np.copy(value_table)
        for state_index in range(env.observation_space.n):
            # Table of q values for this state
            value_table[state_index] = max(
                sum(prob*(reward + DISCOUNT_FACTOR*old_value_table[next_state])
                    for prob, next_state, reward, _ in env.env.P[state_index][action])
                for action in range(env.action_space.n))
        # We will check whether we have reached the convergence i.e whether the difference 
        # between our value table and updated value table is very small.
        # But how do we know it is very small?
        # We set some threshold and then we will see if the difference is less
        # than our threshold, if it is less, we break the loop and return the value function as optimal
        # value function.
        if np.sum(np.fabs(value_table - old_value_table)) <= STATE_VALUE_TABLE_DELTA_THRESHOLD_FOR_CONVERGENCE:
            print ('Value-iteration converged at iteration number: {}.'.format(iter_index + 1))
            break

    return value_table

def extract_policy(env, value_table):
    # Extract the policy as an array with indices being indexes of states.
    # policy[s] is the action that should be taken in state with index s
    # NOTE: We could have computed the policy as part of the value iteration,
    # but we wanted the algorithm to match as close as possible with typical value iteration.
    policy = np.zeros(env.observation_space.n, dtype=int)
    for state_index in range(env.observation_space.n):
        # Initialize the Q values for a state
        q_values = [
            sum(prob*(reward + DISCOUNT_FACTOR*value_table[next_state])
                for prob, next_state, reward, _ in env.env.P[state_index][action])
            for action in range(env.action_space.n)
        ]
        # Select the action which has maximum Q value as an optimal action of the state
        policy[state_index] = np.argmax(q_values)

    return policy

if __name__ == "__main__":
    main()