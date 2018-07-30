import numpy as np
import sys
from lib.envs.bandits.env import Environment
from lib.envs.bandits.action_space import ActionSpace

class BanditEnv(Environment):
    def __init__(self, num_actions = 10, distribution = "bernoulli", evaluation_seed="387"):
        super(BanditEnv, self).__init__()
        
        self.action_space = ActionSpace(range(num_actions))
        self.distribution = distribution
        
        np.random.seed(evaluation_seed)
        
        self.reward_parameters = None
        if distribution == "bernoulli":
            self.reward_parameters = np.random.rand(num_actions)
        elif distribution == "normal":
            self.reward_parameters = (np.random.randn(num_actions), np.random.rand(num_actions))
        elif distribution == "heavy-tail":
            self.reward_parameters = np.random.rand(num_actions)
        else:
            print("Please use a supported reward distribution", flush = True)
            sys.exit(0)
        
        if distribution != "normal":
            self.optimal_arm = np.argmax(self.reward_parameters)
        else:
            self.optimal_arm = np.argmax(self.reward_parameters[0])
    
    def reset(self):
        self.is_reset = True
        return None
    
    def compute_gap(self, action):
        if self.distribution != "normal":
            gap = self.reward_parameters[self.optimal_arm] - self.reward_parameters[action]
        else:
            gap = self.reward_parameters[0][self.optimal_arm] - self.reward_parameters[0][action]
        return gap
    
    def step(self, action):
        self.is_reset = False
        
        valid_action = True
        if (action is None or action < 0 or action >= self.action_space.n):
            print("Algorithm chose an invalid action; reset reward to -inf", flush = True)
            reward = float("-inf")
            gap = float("inf")
            valid_action = False
        
        if self.distribution == "bernoulli":
            if valid_action:
                reward = np.random.binomial(1, self.reward_parameters[action])
                gap = self.compute_gap(action)
        elif self.distribution == "normal":
            if valid_action:
                reward = self.reward_parameters[0][action] + self.reward_parameters[1][action] * np.random.randn()
                gap = self.compute_gap(action)
        elif self.distribution == "heavy-tail":
            if valid_action:
                reward = self.reward_parameters[action] + np.random.standard_cauchy()
                gap = self.compute_gap(action)
        else:
            print("Please use a supported reward distribution", flush = True)
            sys.exit(0)
            
        return(None, reward, self.is_reset, '')
        

