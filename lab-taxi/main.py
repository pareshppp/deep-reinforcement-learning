from agent import Agent
from monitor import interact
from tqdm.auto import tqdm
import gym
import numpy as np

env = gym.make('Taxi-v3')

## Q-Learning
agent = Agent(alpha=0.1, gamma=0.99)
avg_rewards, best_avg_reward = interact(env, agent, is_qlearning=True)

## Expected Sarsa
# agent = Agent(epsilon=0.001, alpha=0.1, gamma=0.99)
# avg_rewards, best_avg_reward = interact(env, agent, is_qlearning=False)