import os
import sys
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import text_flappy_bird_gym
from gymnasium.wrappers.time_limit import TimeLimit
from tqdm import tqdm

from policy import MonteCarlo, Sarsa

NB_EPISODES = 4_000
EVAL_EVERY = 100


def plot_heatmap(data, title):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", color="r")
    plt.title(title)
    fig.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()


if __name__ == "__main__":
    # Init environment. We put a max limit in case our agent finds the optimal strategy.
    env = TimeLimit(
        gym.make("TextFlappyBird-v0", height=15, width=20, pipe_gap=4), 1_000
    )
    # Init agent
    agent = MonteCarlo(env.observation_space, env.action_space, gamma=0.95)
    all_cumul_rewards = []
    for incr in tqdm(range(NB_EPISODES)):
        # initiate environment
        obs, _ = env.reset()
        done = False
        while not done:
            # Select next action
            action = agent.step(obs, random_action=True)
            # Appy action and return new observation of the environment
            next_obs, reward, done, truncated, info = env.step(action)
            agent.remember(obs, action, reward, done or truncated, next_obs)
            obs = next_obs

        if incr % EVAL_EVERY == 0:
            # reset environment
            obs, _ = env.reset()
            cumul_reward = 0
            # iterate
            while True:
                # Select next action
                action = agent.step(obs)

                # Appy action and return new observation of the environment
                obs, reward, done, truncated, info = env.step(action)
                cumul_reward += reward

                # Render the game
                # os.system("clear")
                # sys.stdout.write(env.render())
                # time.sleep(0.2)  # FPS
                # If player is dead break
                if truncated or done:
                    break
            all_cumul_rewards.append(cumul_reward)

    # Plot results
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/" + agent.name):
        os.makedirs("results/" + agent.name)
    folder = "results/" + agent.name + "/"
    plt.plot(np.arange(0, NB_EPISODES, EVAL_EVERY), all_cumul_rewards)
    plt.savefig(folder + "cumul_rewards.png")
    plt.show()
    plot_heatmap(agent.Q.max(axis=2), folder + "Q max")
    plot_heatmap(agent.Q.min(axis=2), folder + "Q min")
    if isinstance(agent, MonteCarlo):
        plot_heatmap(agent.count.max(axis=2), folder + "Count max")
        plot_heatmap(agent.count.min(axis=2), folder + "Count min")

    # Save agent
    agent.save()
    # Test load
    agent.load()

    # reset environment
    obs, _ = env.reset()

    # iterate
    while True:
        # Select next action
        action = agent.step(obs)

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)

        # Render the game
        os.system("clear")
        sys.stdout.write(env.render())
        # time.sleep(0.2)  # FPS
        # If player is dead break
        if done:
            break

    env.close()
