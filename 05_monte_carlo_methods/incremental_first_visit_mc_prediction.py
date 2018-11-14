import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()


def mc_first_visit_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
   Incremental First-Visit Monte Carlo State Value Function Prediction

    Args:
        policy: fn, maps an observation to action probabilities.
        env: gym env, OpenAI gym environment
        num_episodes: int, number of episodes to sample.
        discount_factor: float, gamma discount factor.

    Returns:
        dict, maps from state -> value.
        The state is a tuple and the value is a float.
    """

    def generate_episode(env, policy):
        """
            Generate an `episode` using `policy`
        """
        episode = []  # a list of state, action, reward items
        state = env.reset()
        # loop until a terminal state is reached
        while True:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append(state)
            episode.append(action)
            episode.append(reward)
            if done:
                break
            state = next_state
        return episode

    returns_sum = defaultdict(int)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    # Repeat forever (or for `num_episodes` times)
    for e in range(1, num_episodes + 1):
        # Generate an episode using `policy`
        episode = generate_episode(env, policy)
        states = set()
        # Loop for each time step of episode
        for state_idx in range(0, len(episode), 3):  # states in an episode come every 3 items
            state = episode[state_idx]
            # If store not visited yet in this episode, store it
            if state not in states:
                states.add(state)
                # Increment visits for this state
                returns_count[state] += 1
                # Increment sum of returns
                # by going until the end of the episode and summing up every reward along the way
                returns_sum[state] += sum([discount_factor ** i * episode[reward_idx] for i, reward_idx in
                                           enumerate(range(state_idx + 2, len(episode), 3))])
                # update state value by averaging the returns over the total time `state` was encountered
                V[state] = returns_sum[state] / returns_count[state]
    return V


def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


if __name__ == '__main__':
    V_10k = mc_first_visit_prediction(sample_policy, env, num_episodes=10000)
    plotting.plot_value_function(V_10k, title="10k Steps")
    V_500k = mc_first_visit_prediction(sample_policy, env, num_episodes=500000)
    plotting.plot_value_function(V_500k, title="500k Steps")
