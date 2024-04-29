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





def plot_results(result_array, wind_proportions, n_planning_updates, name="plot", baseline=None, labels=None):
    plot = LearningCurvePlot()
    for i, wind in enumerate(wind_proportions):
        for j, n_updates in enumerate(n_planning_updates):
            if labels is None:
                lbl = f"wind: {wind}, n_updates: {n_updates}"
            else:
                lbl = labels[i + j]
            plot.add_curve(x=range(len(result_array[0, 0, :])), y=smooth(result_array[i, j], window=10),
                           label=lbl)
        if baseline is not None:
            plot.add_curve(x=range(len(result_array[0, 0, :])), y=smooth(baseline[i, 0], window=10),
                           label=f"QLearning, wind: {wind}")
    plot.save(name=f"{name}.png")


def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 10
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1

    wind_proportions = [0.9, 1.0]
    n_planning_updatess = [1, 3, 5]

    max_episode_length = 100

    def run_repetitions(n_repetitions, n_timesteps, eval_interval, gamma, learning_rate, epsilon, wind_proportions, n_planning_updates, max_episode_length, model="Dyna"):
        results = np.zeros((len(wind_proportions), len(n_planning_updates), n_timesteps // eval_interval))

        for windex, wind in enumerate(wind_proportions):
            for planex, plan in enumerate(n_planning_updates):
                for rep in range(n_repetitions):
                    env = WindyGridworld()
                    if model == 'Dyna':
                        agent =  DynaAgent(env.n_states, env.n_actions, learning_rate, gamma)
                    else: 
                        agent = PrioritizedSweepingAgent(env.n_states, env.n_actions, learning_rate, gamma)

                    s = env.reset()
                    rewards = []

                    for step in range(n_timesteps):
                        if (step + 1) % eval_interval == 0:
                            eval = agent.evaluate(env, max_episode_length=max_episode_length)
                            rewards.append(eval )

                        a = agent.select_action(s, epsilon)
                        s_prime, r, done = env.step(a)
                        agent.update(s, a, r, done, s_i, plan)
                        if done:
                            s = env.reset()
                        else:
                            s = s_i

                    results[windex, planex, :] += np.array(rewards) / n_repetitions

        return results

    results = run_repetitions(n_repetitions, n_timesteps, eval_interval, gamma, learning_rate, epsilon,
                              [0], [1], max_episode_length, "Dyna")
    plot_results(results, [0], [1], name="1b")

    # 1c
    for wind in wind_proportions:
        q_learning_baseline = run_repetitions(n_repetitions, n_timesteps, eval_interval, gamma, learning_rate, epsilon,
                                              [wind], [0], max_episode_length, "Dyna")
        results = run_repetitions(n_repetitions, n_timesteps, eval_interval, gamma, learning_rate, epsilon,
                                  [wind], n_planning_updatess, max_episode_length, "Dyna")
        plot_results(results, [wind], n_planning_updatess, name=f"1c_wind{wind}", baseline=q_learning_baseline)

    # 2b
    results = run_repetitions(n_repetitions, n_timesteps, eval_interval, gamma, learning_rate, epsilon,
                              [0], [1], max_episode_length, "SW")
    plot_results(results, [0], [1], name="2b")

    # 2c
    for wind in wind_proportions:
        q_learning_baseline = run_repetitions(n_repetitions, n_timesteps, eval_interval, gamma, learning_rate, epsilon,
                                              [wind], [0], max_episode_length, "Dyna")
        results = run_repetitions(n_repetitions, n_timesteps, eval_interval, gamma, learning_rate, epsilon,
                                  [wind], n_planning_updatess, max_episode_length, "SW")
        plot_results(results, [wind], n_planning_updatess, name=f"2c_wind{wind}", baseline=q_learning_baseline)

    # 3
    for wind in wind_proportions:
        q_learning_baseline = run_repetitions(n_repetitions, n_timesteps, eval_interval, gamma, learning_rate, epsilon,
                                              [wind], [0], max_episode_length, "Dyna")

        results = run_repetitions(n_repetitions, n_timesteps, eval_interval, gamma, learning_rate, epsilon,
                                  [wind], n_planning_updatess, max_episode_length, "Dyna")
        best_dyna_index = np.argmax(np.mean(results[0], axis=1))
        best_performing_dyna = results[0][best_dyna_index]

        results = run_repetitions(n_repetitions, n_timesteps, eval_interval, gamma, learning_rate, epsilon,
                                  [wind], n_planning_updatess, max_episode_length, "SW")
        best_sw_index = np.argmax(np.mean(results[0], axis=1))
        best_performing_sw = results[0][best_sw_index]

        best_performers = np.stack((best_performing_dyna, best_performing_sw))
        best_performers = best_performers.reshape(1, best_performers.shape[0], best_performers.shape[1])

        plot_results(best_performers, [wind],
                     [n_planning_updatess[best_dyna_index], n_planning_updatess[best_sw_index]], name=f"3_wind{wind}",
                     baseline=q_learning_baseline,
                     labels=[f"Best performing Dyna with n_updates: {n_planning_updatess[best_dyna_index]}",
                             f"Best performing SW with n_updates: {n_planning_updatess[best_sw_index]}"])


if __name__ == '__main__':
    experiment()
