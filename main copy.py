import numpy as np
import matplotlib.pyplot as plt
import optuna
from env import GridworldEnv

def get_action(greedy_action, epsilon, n_actions):
    prob = np.random.rand()
    if prob < epsilon:
        return np.random.choice(n_actions)
    else:
        return greedy_action


class TabularRLAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.n_states=n_states
        self.n_actions=n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = np.zeros((self.n_states, self.n_actions))
        self.policy = np.zeros(self.n_states, dtype=int)



    def update_policy(self):
        for s in range(self.n_states):
            self.policy[s] = np.argmax(self.Q[s])



    def train_q_learning_episode(self, env, max_steps=200):
        s,_ = env.reset()
        done=False
        step_i=0
        total_reward=0.0

        while done == 'False' and step_i < max_steps:
            self.update_policy()
            a= get_action(self.policy[s], self.epsilon, self.n_actions)

            s_prime, rwd,done, _, _ = env.step(a)
            total_reward +=rwd

            future_q = 0.0 if done =='True' else np.max(self.Q[s_prime])
            td_target = rwd + self.gamma * future_q
            td_error = td_target - self.Q[s][a]
            self.Q[s][a] += self.alpha * td_error

            s=s_prime
            step_i +=1

        return total_reward


    
