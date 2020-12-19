import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon=0.1, alpha=0.01, gamma=1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.random_sample() > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            ## Q-Learning
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
            ## Expected Sarsa
            # self.Q[state][action] += self.alpha * (reward + self.gamma * self.expected_Q(state) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
        
    def expected_Q(self, state):
        policy_s = np.ones(self.nA) * (self.epsilon / self.nA)
        best_action = np.argmax(self.Q[state])
        policy_s[best_action] = 1 - self.epsilon + (self.epsilon / self.nA)
        expected_Q_value = np.dot(self.Q[state], policy_s)
        return expected_Q_value