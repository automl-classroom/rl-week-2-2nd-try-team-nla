from __future__ import annotations

import gymnasium as gym
import numpy as np


# ------------- TODO: Implement the following environment -------------
class MyEnv(gym.Env):
    """
    Simple 2-state, 2-action environment with deterministic transitions.

    Actions
    -------
    Discrete(2):
    - 0: move to state 0
    - 1: move to state 1

    Observations
    ------------
    Discrete(2): the current state (0 or 1)

    Reward
    ------
    Equal to the action taken.

    Start/Reset State
    -----------------
    Always starts in state 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initializes the observation and action space for the environment."""
        # self.rng = np.random.default_rng(seed)

        self.number_of_actions = 0
        self.rewards = 0

        self.state = 0
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

        self.states = np.arange(2)
        self.actions = np.arange(2)

        self.transition_matrix = self.T = self.get_transition_matrix()

    def reset(self, seed: int):
        self.rewards = 0
        self.number_of_actions = 0
        self.state = 0
        return self.state, {}  # I do not know why {}

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")
        self.state = action
        self.number_of_actions += 1
        reward = float(self.number_of_actions)
        terminated = False
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def get_reward_per_action(self):
        nS, nA = self.observation_space.n, self.action_space.n
        reward_matrix = np.ones((nS, nA), dtype=float)
        return reward_matrix

    def get_transition_matrix(self):
        nS, nA = self.observation_space.n, self.action_space.n
        reward_matrix_depending_states = np.ones((nS, nA, nS), dtype=float)
        return reward_matrix_depending_states


class PartialObsWrapper(gym.Wrapper):
    """Wrapper that makes the underlying env partially observable by injecting
    observation noise: with probability `noise`, the true state is replaced by
    a random (incorrect) observation.

    Parameters
    ----------
    env : gym.Env
        The fully observable base environment.
    noise : float, default=0.1
        Probability in [0,1] of seeing a random wrong observation instead
        of the true one.
    seed : int | None, default=None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: gym.Env, noise: float = 0.1, seed: int | None = None):
        super().__init__(env)
        assert 0.0 <= noise <= 1.0, "noise must be in [0, 1]"
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed: int):
        true_obs, info = self.env.reset(seed=seed), {}
        return self._noisy_obs(true_obs), info

    def step(self, action: int):
        true_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._noisy_obs(true_obs), reward, terminated, truncated, info

    def _noisy_obs(self, true_obs: int) -> int:
        if self.rng.random() < self.noise:
            # Since only 2 states (0 and 1), just flip the value
            return 1 - true_obs
        return true_obs
