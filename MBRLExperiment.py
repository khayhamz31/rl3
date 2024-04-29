#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
import os 
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 10
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1
    
    wind_proportions=[0.9,1.0]
    n_planning_updates = [1,3,5] 

    def run_repetitions():
        results = np.zeros((len(wind_proportions),len(n_planning_updates),n_repetitions,n_timesteps//eval_interval))
        for windex, wind in enumerate(wind_proportions):
            for plandex, plan in enumerate(n_planning_updates):
                for rep in range(n_repetitions):
                    env = WindyGridworld(wind_proportion=wind)
                    agent = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma)
                    current = env.reset()
                    returns = np.zeros(n_timesteps//eval_interval)
                    for idx in range(n_timesteps):
                        if (idx + 1) % eval_interval == 0:
                            eval = agent.evaluate(env)
                            index = (idx + 1) // eval_interval - 1 
                            returns[index] += eval 
                        act = agent.select_action(current,epsilon)
                        s_next,r,done = env.step(act)
                        agent.update(s=current,a=act,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates[0])
                        if done:
                            current = env.reset()
                        else:
                            current = s_next
                    results[windex][plandex][rep] = returns 
            return results 

    def graphs():
        lab = ['stochastic','deterministic']
        if not os.path.exists('results.npy'):
            results = run_repetitions()
            np.save("results.npy",results)
        else: 
            print('tada')
            results = np.load("results.npy")
        averaged = np.mean(results,axis = 2)
        for i in range(results.shape[0]):
            LCTest = LearningCurvePlot(title= f"{lab[i]} curves")
            for j,each in enumerate(n_planning_updates):
                smoothed = smooth(averaged[i][j],window = 31)
                LCTest.add_curve(np.arange(results.shape[3]),smoothed, label = f'{each} planning updates')
            LCTest.save(name = f'{lab[i]} curves')
    
    graphs()

if __name__ == '__main__': 
    experiment()
