import numpy as np
import pprint
import sys

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv
from DynamicProgramming.policy_evaluation_two_arrays import policy_eval

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    General Policy Iteration algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

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

    # Initialization
    # Start with random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:

        #######################
        # Policy evaluation
        #######################
        V = policy_eval(policy=policy, env=env)

        #######################
        # Policy improvement
        #######################
        policy_stable = True
        for s in range(env.nS):
            # choose current action by acting greedily on current policy
            chosen_a = np.argmax(policy[s])
            action_values = one_step_lookahead(s, V)
            # act greedily wrt to current value function
            best_a = np.argmax(action_values)
            policy[s] = np.eye(env.nA)[best_a]
            # if policy does not change anymore, it converged to optimal
            if best_a != chosen_a:
                policy_stable = False
        if policy_stable:
            return (policy, V)


if __name__ == '__main__':
    #################################
    ### Example 4.1 4x4 Gridworld
    #################################
    policy, v = policy_improvement(env)
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
