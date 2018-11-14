import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and environment's dynamics.

    Args:
        policy: ndarray, [S x A] matrix representing the policy.
        env: OpenAI env, env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
            discount_factor: float, Gamma discount factor (default: 1.0, i.e. undiscounted)
        theta: float, we stop evaluation once our value function change is less than theta for all states.

    Returns: ndarray, vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0.
        for s, actions in env.P.items():
            v = 0.  # store the next-time-step value for state s
            for a in actions.keys():
                for prob, next_state, reward, done in env.P[s][a]:
                    v += policy[s, a] * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v  # modify value function in-place
        if delta < theta:
            break
    return V


if __name__ == '__main__':

    #################################
    ### Example 4.1 4x4 Gridworld
    #################################
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v = policy_eval(random_policy, env)

    expected_v = np.array(
        [
            0, -14, -20, -22,
            -14, -18, -20, -20,
            -20, -20, -18, -14,
            -22, -20, -14, 0
        ]
    )

    print('ANSWER: {}'.format(np.round(v, decimals=0)))
    print('SOLUTION: {}'.format(expected_v))

    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)