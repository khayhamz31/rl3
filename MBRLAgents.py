#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        self.transition = np.zeros((n_states,n_actions,n_states))
        self.rewards = np.zeros((n_states,n_actions,n_states))
        
    def select_action(self, s, epsilon):
        best_action = np.argmax(self.Q_sa[s])
        probs = np.zeros(self.n_actions)
        for idx in range(self.n_actions):
            if idx == best_action:
                probs[idx] = 1 - epsilon
            else:
                probs[idx] = epsilon/(self.n_actions-1)
        total_prob = sum(probs)
        probs /= total_prob
        a = np.random.choice(self.n_actions,p=probs)
        return a 
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        # update transition counts
        self.transition[s][a][s_next] += 1
        # update reward sum
        self.rewards[s][a][s_next] += r
        # transition function 
        transition = np.zeros((self.n_states,self.n_actions,self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                transitions = np.sum(self.transition[s][a])
                if transitions > 0:
                    for s1 in range(self.n_states):
                        if self.transition[s][a][s1] > 0:
                            transition[s][a][s1] = self.transition[s][a][s1] / transitions
        # reward function
        reward = np.zeros((self.n_states,self.n_actions,self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                    for s1 in range(self.n_states):
                        if self.transition[s][a][s1]>0:
                            reward[s][a][s1] = self.rewards[s][a][s1] / self.transition[s][a][s1]
        # update Q table
        self.Q_sa[s][a] += self.learning_rate * (r + self.gamma * (max(self.Q_sa[s_next]))- self.Q_sa[s][a])
        for _ in range(n_planning_updates):
            # random state where n(s) > 0
            state = np.random.choice(np.where(np.sum(self.transition, axis=(1, 2)) > 0)[0])
            # random action where n(s,a) > 0
            action = np.random.choice(np.where(self.transition[state].sum(axis=1) > 0)[0])
            # simulate the model based on the estimates 
            new_state = np.random.choice(range(self.n_actions),p=transition[state][action])
            new_r = reward[state][action][new_state]
            # update Q table
            self.Q_sa[state][action] += self.learning_rate * (new_r + (self.gamma * max(self.Q_sa[new_state])) - self.Q_sa[state][action])


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, max_queue_size=200, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.Q_sa = np.zeros((n_states,n_actions))
        self.transition = np.zeros((n_states, n_actions, n_states))
        self.rewards = np.zeros((n_states, n_actions, n_states))
        self.queue = PriorityQueue()
        self.max_queue_size = max_queue_size
        
    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection
        best_action = np.argmax(self.Q_sa[s])
        probs = np.zeros(self.n_actions)
        for idx in range(self.n_actions):
            if idx == best_action:
                probs[idx] = 1 - epsilon
            else:
                probs[idx] = epsilon/(self.n_actions-1)
        total_prob = sum(probs)
        probs /= total_prob
        a = np.random.choice(self.n_actions,p=probs)
        return a 
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        
        # TO DO: Add Prioritized Sweeping code
         # Update transition counts
        self.transition[s][a][s_next] += 1
        # Update reward sum
        self.rewards[s][a][s_next]+= r

        transition = np.zeros((self.n_states,self.n_actions,self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                transitions = np.sum(self.transition[s][a])
                if transitions > 0:
                    for s1 in range(self.n_states):
                        if self.transition[s][a][s1] > 0:
                            transition[s][a][s1] = self.transition[s][a][s1] / transitions
        # reward function
        reward = np.zeros((self.n_states,self.n_actions,self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                    for s1 in range(self.n_states):
                        if self.transition[s][a][s1]>0:
                            reward[s][a][s1] = self.rewards[s][a][s1] / self.transition[s][a][s1]
        p_score = np.abs(r + self.gamma * max(self.Q_sa[s_next])-self.Q_sa[s][a])

        if p_score > self.priority_cutoff and self.queue.qsize() < self.max_queue_size:
            self.queue.put((-p_score,(s,a)))
        
        for _ in range(n_planning_updates):
            score, pair = self.queue.get()
            state = pair[0]
            action = pair[1]
            new_state = np.random.choice(range(self.n_actions),p=transition[state][action])
            new_r = reward[state][action][new_state]
            self.Q_sa[state][action] += self.learning_rate * (new_r + (self.gamma * max(self.Q_sa[new_state])-self.Q_sa[state][action]))
            for i in range(self.n_states):
                for act in range(self.n_actions):
                    for i_i in range(self.n_states):
                        if i_i == state and self.transition[i][act][i_i] > 0:
                            r_1 = self.rewards[i][act][i_i]
                            p_1 = np.abs(r_1 + self.gamma * max(self.Q_sa[state])-self.Q_sa[i][act])
                            if p_1 > self.priority_cutoff and self.queue.qsize() < self.max_queue_size:
                                self.queue.put((-p_1,(i,act)))


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return        

def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'dyna' # or 'ps' 
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
            
    
if __name__ == '__main__':
    test()
