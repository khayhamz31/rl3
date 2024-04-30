#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue, Empty
from MBRLEnvironment import WindyGridworld


class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))  # intialise the q table
        self.transition = np.zeros((n_states, n_actions, n_states))  # transition count
        self.rewards = np.zeros((n_states, n_actions, n_states))  # reward sum
        self.transition_estimate = np.zeros((self.n_states, self.n_actions, self.n_states))  # transition estimate
        self.reward_estimate = np.zeros((self.n_states, self.n_actions, self.n_states))  # reward estimate

    def select_action(self, s, epsilon):
        if np.random.rand() < epsilon:
            a = np.random.randint(self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s])
        return a

    def update(self, s, a, r, done, s_next, n_planning_updates):
        # Update model counts and rewards
        self.transition[s, a, s_next] += 1
        self.rewards[s, a, s_next] += r

        # Track predecessors for each state
        self.predecessors[s_next].add((s, a))

        # Calculate priority
        max_q_next = np.max(self.Q_sa[s_next])
        priority = np.abs(r + self.gamma * max_q_next - self.Q_sa[s, a])

        # Check if priority is above threshold before inserting into priority queue
        if priority > self.priority_cutoff:
            self.queue.put((-priority, (s, a)))  # Negative priority because PriorityQueue is a min-heap

        # Planning step
        for _ in range(n_planning_updates):
            if self.queue.empty():
                break

            # Get the highest priority state-action pair
            _, (s_pri, a_pri) = self.queue.get()

            # Find the most likely next state and reward based on model counts and rewards
            sampled_states = np.where(self.transition[s_pri, a_pri] > 0)[0]
            transition_prob = self.transition[s_pri, a_pri, sampled_states] / np.sum(self.transition[s_pri, a_pri])
            next_state = np.random.choice(sampled_states, p=transition_prob)
            reward = self.rewards[s_pri, a_pri, next_state] / self.transition[s_pri, a_pri, next_state]

            # Update Q-value for the selected state-action pair
            self.Q_sa[s_pri, a_pri] += self.learning_rate * (
                reward + self.gamma * np.max(self.Q_sa[next_state]) - self.Q_sa[s_pri, a_pri]
            )

            # Update priorities for all predecessors of the selected state
            for sp, ap in self.predecessors[s_pri]:
                if np.sum(self.transition[sp, ap]) > 0:
                    reward_pred = self.rewards[sp, ap, s_pri] / np.sum(self.transition[sp, ap])
                    max_q_pred = np.max(self.Q_sa[s_pri])
                    priority_pred = np.abs(reward_pred + self.gamma * max_q_pred - self.Q_sa[sp, ap])

                    # Check if priority is above threshold before inserting into priority queue
                    if priority_pred > self.priority_cutoff:
                        self.queue.put((-priority_pred, (sp, ap)))

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])  # greedy action selection
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
        self.Q_sa = np.zeros((n_states, n_actions))  # initialise q_table
        self.transition = np.zeros((n_states, n_actions, n_states))  # intialise transtiion counts
        self.rewards = np.zeros((n_states, n_actions, n_states))  # initialise cumulative rewards
        self.queue = PriorityQueue()  # initalise priority queue
        self.max_queue_size = max_queue_size  # initialise max queue
        self.transition_estimate = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.reward_estimate = np.zeros((self.n_states, self.n_actions, self.n_states))

    def select_action(self, s, epsilon):
        if np.random.rand() < epsilon:
            a = np.random.randint(self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s])
        return a

    def update(self, s, a, r, done, s_next, n_planning_updates):

        # TO DO: Add Prioritized Sweeping code
        self.trans_counts[s, a, s_next] += 1
        self.reward_sum[s, a, s_next] += r
        # Helper code to work with the queue
        # Put (s,a) on the queue with priority p (needs a minus since the queue pops the smallest priority first)
        p = abs(r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s, a])
        if p > self.priority_cutoff:
            self.queue.put((-p, (s, a)))

        for _ in range(n_planning_updates):
            # Retrieve the top (s,a) from the queue
            try:
                _, (s, a) = self.queue.get(False)  # get the top (s,a) for the queue
            except Empty:
                break

            sampled_states = np.where(self.trans_counts[s, a] > 0)[0]
            trans_func = (self.trans_counts[s, a, sampled_states] / np.sum(self.trans_counts[s, a]))

            next_state = np.random.choice(sampled_states, p=trans_func)
            reward = self.reward_sum[s, a, next_state] / self.trans_counts[s, a, next_state]

            self.Q_sa[s, a] += (self.learning_rate *
                                (reward + self.gamma * np.max(self.Q_sa[next_state]) - self.Q_sa[s, a]))

            prev_states, prev_actions = np.where(self.trans_counts[:, :, s] > 0)
            for i in range(len(prev_states)):
                s_b, a_b = prev_states[i], prev_actions[i]
                r_b = self.reward_sum[s_b, a_b, s] / self.trans_counts[s_b, a_b, s]
                p = abs(r_b + self.gamma * np.max(self.Q_sa[s]) - self.Q_sa[s_b, a_b])
                if p > self.priority_cutoff:
                    self.queue.put((-p, (s_b, a_b)))

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])  # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])  # greedy action selection
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

    policy = 'dyna'
    # policy = 'ps'
    # policy = 'dyna' or 'ps'
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001

    # Initialize environment and policy
    env = WindyGridworld()
    print(env.n_states)
    if policy == 'dyna':
        pi = DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)  # Initialize Dyna policy
    elif policy == 'ps':
        pi = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma)  # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))

    # Prepare for running
    s = env.reset()
    continuous_mode = False

    for t in range(n_timesteps):
        # Select action, transition, update policy
        a = pi.select_action(s, epsilon)
        s_next, r, done = env.step(a)
        pi.update(s=s, a=a, r=r, done=done, s_next=s_next, n_planning_updates=n_planning_updates)

        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=plot_optimal_policy,
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
