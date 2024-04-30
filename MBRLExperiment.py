#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth
import os 


def run_repetitions(age,wind_proportions,n_planning_updates,n_repetitions,n_timesteps,eval_interval,learning_rate,gamma,epsilon):
    results = np.zeros((len(wind_proportions),len(n_planning_updates),n_repetitions,n_timesteps//eval_interval))
    for windex, wind in enumerate(wind_proportions):
        for plandex, plan in enumerate(n_planning_updates):
            for rep in range(n_repetitions):
                env = WindyGridworld(wind_proportion=wind)
                if age == 'd':
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
                    agent.update(s=current,a=act,r=r,done=done,s_next=s_next,n_planning_updates=plan)
                    if done:
                        current = env.reset()
                    else:
                        current = s_next
                results[windex][plandex][rep] = returns 
    np.save(f'{age}_{wind_proportions[0]}.npy',results)
    return results

def run_repetitions_ps(n_timesteps, n_repetitions, eval_interval, gamma, learning_rate, epsilon, n_planning_update, wind_proportion):
    result = np.zeros((n_repetitions,n_timesteps//eval_interval))
    for reps in range(n_repetitions): 
        env = WindyGridworld(wind_proportion=wind_proportion[0])
        agent = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma,)
        current = env.reset()
        returns = np.zeros(n_timesteps//eval_interval)
        for idx in range(n_timesteps):
            if (idx + 1) % eval_interval == 0:
                eval = agent.evaluate(env)
                index = (idx + 1) // eval_interval - 1 
                returns[index] += eval 
            act = agent.select_action(current,epsilon)
            s_next,r,done = env.step(act)
            agent.update(s=current,a=act,r=r,done=done,s_next=s_next,n_planning_updates=plan)
            if done:
                current = env.reset()
            else:
                current = s_next
        result[reps] = returns
def graphs(results, agent_type, env_type,updates,n_timesteps,eval_interval):
        LCP = LearningCurvePlot()
        mean  = np.mean(results,axis=2)
        if agent_type == 'd':
            agent_t = 'dyna'
        else: 
            agent_t = 'Prioritized sweeping'
        if env_type == 0:
            evn_t = 'stochastic'
        else: 
            evn_t = 'deterministic'
        for idx,update in enumerate(updates):
            smoothed = smooth(mean[0][idx],window = 11)
            LCP.add_curve(range(n_timesteps//eval_interval), smoothed, label=f"{evn_t} environment, updates: {update}")
        LCP.save(f'{agent_t},{evn_t}')

def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 10
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1

    wind_proportions=[0.9,1.0]
    n_planning_updates = [0,1,3,5] 

    # dd = run_repetitions('d',[1],n_planning_updates = n_planning_updates,n_repetitions = n_repetitions ,n_timesteps = n_timesteps ,eval_interval = eval_interval,learning_rate = learning_rate,gamma = gamma,epsilon = epsilon)
    # ds = run_repetitions('d',[0.9],n_planning_updates = n_planning_updates,n_repetitions = n_repetitions ,n_timesteps = n_timesteps ,eval_interval = eval_interval,learning_rate = learning_rate,gamma = gamma,epsilon = epsilon)
    pd = run_repetitions('ps',[1],n_planning_updates = n_planning_updates,n_repetitions = n_repetitions ,n_timesteps = n_timesteps ,eval_interval = eval_interval,learning_rate = learning_rate,gamma = gamma,epsilon = epsilon)
    ps = run_repetitions('ps',[0.9],n_planning_updates = n_planning_updates,n_repetitions = n_repetitions ,n_timesteps = n_timesteps ,eval_interval = eval_interval,learning_rate = learning_rate,gamma = gamma,epsilon = epsilon)
    # dd = np.load("d_1.npy")
    # ds = np.load("d_0.9.npy")
    # pd = np.load("ps_1.npy")
    # ps = np.load("ps_0.9.npy")
    # print(pd)
    # print(ps)
    # print(pd)
    # print(ps)
    # graphs(dd, 'd', 1,n_planning_updates,n_timesteps,eval_interval)
    # graphs(ds, 'd', 0,n_planning_updates,n_timesteps,eval_interval)
    graphs(pd, 'p', 1,n_planning_updates,n_timesteps,eval_interval)
    graphs(ps, 'p', 0,n_planning_updates,n_timesteps,eval_interval)

if __name__ == '__main__':
    experiment()
