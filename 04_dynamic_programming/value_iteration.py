import numpy as np
import pprint
import sys

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(s, V):
        '''
            One step lookahead, Eq. (4.9)
        '''
        action_values = np.zeros(env.nA)
        for a in env.P[s].keys():
            v = 0
            for prob, next_state, reward, done in env.P[s][a]:
                v += prob * (reward + discount_factor * V[next_state])
            action_values[a] = v
        return action_values

    # Initialize value function and random policy
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    while True:
        next_v = np.zeros_like(V)
        for s in range(env.nS):
            action_values = one_step_lookahead(s, V)
            next_v[s] = np.max(action_values)
        # Stop evaluation once change of value funtion is too small otherwise update value function with new approx
        if np.all(np.abs(next_v - V) < theta):
            break
        else:
            V = next_v

    # Output a deterministic policy
    for s in range(env.nS):
        action_values = one_step_lookahead(s, V)
        policy[s] = np.eye(env.nA)[np.argmax(action_values)]

    return policy, V


if __name__ == '__main__':
    policy, v = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    expected_v = np.array(
        [
            0, -1, -2, -3,
            -1, -2, -3, -2,
            -2, -3, -2, -1,
            -3, -2, -1, 0
        ]
    )
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
