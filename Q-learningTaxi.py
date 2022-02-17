# Analysis of Q-learning Exploration-Exploitation Trade-off in Taxi Problem
# Reinforcement Learning Practical
# Authors: Sofia Teeriaho and Mohamed Gamil
# References: The following code was inspired by our previous labs and
# [1] CoderOne - Tutorial: An Introduction to Reinforcement Learning Using OpenAI Gym
# url: https://www.gocoder.one/blog/rl-tutorial-with-openai-gym
# [2] Reinforcement Q-Learning from Scratch in Python with OpenAI Gym
# url: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

import numpy as np
import gym
import random
import matplotlib.pyplot as plt

# Create Taxi environment
env = gym.make('Taxi-v3')

# Function to initialize Q table with 0's
def init_qtable():

    states = env.observation_space.n
    actions = env.action_space.n
    q_table = np.zeros((states, actions))

    return q_table

def e_greedy(epsilon, state, qtable, index):

    random_list = []
    nr = int(epsilon * float(100))
    for i in range(nr):
        random_list.append(random.randint(0, 100))

    if index in random_list:
        # explore
        action = env.action_space.sample()
    else:
        # exploit
        action = np.argmax(qtable[state, :])

    return action

def run_experiments(n_exp, time_steps, epsilon):

    # hyperparameters
    l_rate = 0.8  # learning rate
    discount = 0.8  # discount factor

    qtable = init_qtable()

    nr_steps = []
    total_rewards = []

    for experiment in range(n_exp):

        # reset the environment and return to the initial state
        state = env.reset()

        # keep track of rewards
        total = 0

        for step in range(time_steps):

            action = e_greedy(epsilon, state, qtable, step)

            #action = greedy(state, qtable)

            # Take action
            new_state, reward, end, info = env.step(action)

            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + l_rate * (reward + discount * np.max(qtable[new_state, :]) - qtable[state, action])

            # Update state
            state = new_state

            total += reward

            # Terminate experiment
            if end:
                break

        nr_steps.append(step)
        total_rewards.append(total)

    return qtable, nr_steps, total_rewards

# Final run to determine if agent trained
# Referenced from [1]
def trained_run(qtable, time_steps):

    # reset environment
    state = env.reset()

    total_rewards = 0

    print(f"The following demonstrates an agent after training (e = 0.0)")

    for step in range(time_steps):

        print("Step {}".format(step + 1))

        action = np.argmax(qtable[state, :])
        new_state, reward, end, info = env.step(action)
        total_rewards += reward

        env.render()

        print(f"Cumulative Reward: {total_rewards}")
        state = new_state

        if end:
            break

def plot_values(y_values, epsilon, n):

    x_values = [i for i in range(1000) if i % n == 0]
    plt.plot(x_values, y_values, label=epsilon, markersize=2)

def show():

    plt.figure(1)
    plt.title('Steps per experiment (averaged every 10 experiments)')
    plt.ylabel('Number of steps')
    plt.xlabel('Experiment')
    plt.legend(title="epsilon")
    axes = plt.gca()
    axes.yaxis.grid()

    plt.figure(2)
    plt.title('Cumulative reward per experiment (averaged every 10 experiments)')
    plt.ylabel('Reward value')
    plt.xlabel('Experiment')
    plt.legend(title="epsilon")
    axes = plt.gca()
    axes.yaxis.grid()

    plt.show()

def main():

    # time steps per experiment
    time_steps = 100
    # number of experiments
    n_exp = 1000

    # parameter
    epsilon = [0.0, 0.1, 0.5]

    for i in epsilon:
        qtable, nr_steps, total_rewards = run_experiments(n_exp, time_steps, i)

        # find the average of every n experiments
        n = 10
        averages1 = [sum(nr_steps[i:i + n]) // n for i in range(0, len(nr_steps), n)]
        averages2 = [sum(total_rewards[i:i + n]) // n for i in range(0, len(total_rewards), n)]

        plt.figure(1)
        plot_values(averages1, i, n)

        plt.figure(2)
        plot_values(averages2, i, n)

    qtable, nr_steps, total_rewards = run_experiments(n_exp, time_steps, epsilon[0])

    trained_run(qtable, time_steps)

    show()

    env.close()

if __name__ == '__main__':
    main()