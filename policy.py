import os
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np


class Agent:
    def __init__(self) -> None:
        raise NotImplementedError

    def step(self, obs: np.ndarray, random_action: bool = False) -> int:
        raise NotImplementedError

    def remember(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class MonteCarlo(Agent):
    def __init__(
        self,
        env_observation_space: tuple | gym.Space,
        env_action_space: gym.Space,
        gamma: float = 0.95,
    ) -> None:

        self.gamma = gamma
        self.n_states, self.n_actions = self._get_state_action_size(
            env_observation_space, env_action_space
        )
        # Create memories
        self.Q = np.zeros(self.n_states + (self.n_actions,))
        self.count = np.zeros(self.n_states + (self.n_actions,))
        # Episode history
        self.episode_states: List[np.ndarray] = []
        self.epsiode_actions: List[int] = []
        self.episode_reward: List[float] = []
        # Epsilon
        self.epsilon = 1.0
        self.nb_episodes = 0

    def _get_state_action_size(
        self, env_observation_space: gym.Space, env_action_space: gym.Space
    ) -> Tuple[Tuple[int, ...], int]:
        """Compute the state and action size

        Args:
            env_observation_space (gym.spaces): observation space
            env_action_space (gym.spaces): action space

        Raises:
            NotImplementedError: All cases are not handled in this report

        Returns:
            Tuple[Tuple[int, ...], int]: state size and action size
        """
        if isinstance(env_action_space, gym.spaces.Discrete):
            n_actions = env_action_space.n
        else:
            raise NotImplementedError
        if isinstance(env_observation_space, gym.spaces.Tuple):
            if isinstance(env_observation_space[0], gym.spaces.Discrete):
                n_states = tuple(
                    [state_space.n for state_space in env_observation_space]
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return n_states, n_actions

    def step(self, obs: np.ndarray, random_action: bool = False) -> int:
        if random_action and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        return np.argmax(self.Q[obs])

    def remember(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ) -> None:
        self.episode_states.append(obs)
        self.epsiode_actions.append(action)
        self.episode_reward.append(reward)

        if done:
            # Updatae value function
            cumul_reward = 0.0
            for incr in reversed(range(len(self.episode_states))):
                state = self.episode_states[incr]
                action = self.epsiode_actions[incr]
                cumul_reward = self.gamma * cumul_reward + self.episode_reward[incr]
                self.count[state][action] += 1
                self.Q[state][action] += (1 / self.count[state][action]) * (
                    cumul_reward - self.Q[state][action]
                )

            # Update epsilon
            self.nb_episodes += 1
            self.epsilon = 0.1 + 0.9 * np.exp(-0.001 * self.nb_episodes)

            # Reset episode history
            self.episode_states = []
            self.epsiode_actions = []
            self.episode_reward = []

    def load(self) -> None:
        folder = "models/" + self.name + "/"
        if os.path.exists(folder):
            self.Q = np.load(folder + "Q.npy")
            self.count = np.load(folder + "count.npy")
        else:
            print("No model found")

    def save(self) -> None:
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("models/" + self.name):
            os.makedirs("models/" + self.name)
        folder = "models/" + self.name + "/"
        np.save(folder + "Q.npy", self.Q)
        np.save(folder + "count.npy", self.count)


class Sarsa(Agent):
    def __init__(
        self,
        env_observation_space: tuple | gym.Space,
        env_action_space: gym.Space,
        gamma: float = 0.95,
        lmbda: float = 0.9,
        alpha: float = 0.1,
    ) -> None:

        self.gamma = gamma
        self.lmbda = lmbda
        self.alpha = alpha
        self.n_states, self.n_actions = self._get_state_action_size(
            env_observation_space, env_action_space
        )
        # Create memories
        self.e = np.zeros(self.n_states + (self.n_actions,))
        self.Q = np.zeros(self.n_states + (self.n_actions,))
        # Episode history
        self.episode_states: List[np.ndarray] = []
        self.epsiode_actions: List[int] = []
        self.episode_reward: List[float] = []
        # Epsilon
        self.epsilon = 1.0
        self.nb_episodes = 0

    def _get_state_action_size(
        self, env_observation_space: gym.Space, env_action_space: gym.Space
    ) -> Tuple[Tuple[int, ...], int]:
        """Compute the state and action size

        Args:
            env_observation_space (gym.spaces): observation space
            env_action_space (gym.spaces): action space

        Raises:
            NotImplementedError: All cases are not handled in this report

        Returns:
            Tuple[Tuple[int, ...], int]: state size and action size
        """
        if isinstance(env_action_space, gym.spaces.Discrete):
            n_actions = env_action_space.n
        else:
            raise NotImplementedError
        if isinstance(env_observation_space, gym.spaces.Tuple):
            if isinstance(env_observation_space[0], gym.spaces.Discrete):
                n_states = tuple(
                    [state_space.n for state_space in env_observation_space]
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return n_states, n_actions

    def step(self, obs: np.ndarray, random_action: bool = False) -> int:
        if random_action and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        return np.argmax(self.Q[obs])

    def remember(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ) -> None:
        self.episode_states.append(obs)
        self.epsiode_actions.append(action)
        self.episode_reward.append(reward)

        if done:
            # Update values
            for incr, state in enumerate(self.episode_states):
                if incr == len(self.episode_states) - 1:
                    delta = (
                        self.episode_reward[incr]
                        - self.Q[state][self.epsiode_actions[incr]]
                    )
                else:
                    delta = (
                        self.episode_reward[incr]
                        + self.gamma
                        * self.Q[self.episode_states[incr + 1]][
                            self.epsiode_actions[incr + 1]
                        ]
                        - self.Q[state][self.epsiode_actions[incr]]
                    )
                self.e[state][self.epsiode_actions[incr]] += 1
                self.Q += self.alpha * delta * self.e
                self.e *= self.gamma * self.lmbda
            # Update epsilon
            self.nb_episodes += 1
            self.epsilon = 0.1 + 0.9 * np.exp(-0.0001 * self.nb_episodes)

            # Reset episode history
            self.episode_states = []
            self.epsiode_actions = []
            self.episode_reward = []

    def load(self) -> None:
        folder = "models/" + self.name + "/"
        if os.path.exists(folder):
            self.Q = np.load(folder + "Q.npy")
            self.e = np.load(folder + "e.npy")
        else:
            print("No model found")

    def save(self) -> None:
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("models/" + self.name):
            os.makedirs("models/" + self.name)
        folder = "models/" + self.name + "/"
        np.save(folder + "Q.npy", self.Q)
        np.save(folder + "e.npy", self.e)
