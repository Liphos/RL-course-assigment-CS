import os
import sys
import time

import gymnasium as gym
import text_flappy_bird_gym

if __name__ == "__main__":

    # initiate environment
    env = gym.make("TextFlappyBird-v0", height=15, width=20, pipe_gap=4)
    obs = env.reset()

    # iterate
    while True:

        # Select next action
        action = (
            env.action_space.sample()
        )  # for an agent, action = agent.policy(observation)

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)
        # Render the game
        os.system("clear")
        sys.stdout.write(env.render())
        time.sleep(2)  # FPS

        # If player is dead break
        if done:
            break

    env.close()
