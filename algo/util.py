import os
from typing import Optional

import time
import numpy as np

import gym
from gym import spaces
from gym.envs import register
from minatar import Environment

from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper


class BaseEnv(gym.Env):
    # Adapted from https://github.com/qlan3/gym-games
    metadata = {"render.modes": ["human", "array"]}

    def __init__(self, game, display_time=50, use_minimal_action_set=False, max_steps=500000, **kwargs):
        self.game_name = game
        self.display_time = display_time

        self.game_kwargs = kwargs
        self.seed()

        if use_minimal_action_set:
            self.action_set = self.game.minimal_action_set()
        else:
            self.action_set = list(range(self.game.num_actions()))

        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=self.game.state_shape(), dtype=bool
        )

        self.max_steps = max_steps
        self.elapsed_steps = 0

    def step(self, action):
        action = self.action_set[action]
        reward, done = self.game.act(action)
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_steps:
            done = True
        return self.game.state(), reward, done, {}

    def reset(self):
        self.elapsed_steps = 0
        self.game.reset()
        return self.game.state()

    def seed(self, seed=None):
        self.game = Environment(
            env_name=self.game_name,
            random_seed=seed,
            **self.game_kwargs
        )
        return seed

    def render(self, mode="human"):
        if mode == "array":
            return self.game.state()
        elif mode == "human":
            self.game.display_state(self.display_time)

    def close(self):
        if self.game.visualized:
            self.game.close_display()
        return 0


def register_envs():
    for game in ["asterix", "breakout", "freeway", "seaquest", "space_invaders"]:
        name = game.title().replace('_', '')
        register(
            id="{}-MinAtar-v0".format(name),
            entry_point="algo.util:BaseEnv",
            kwargs=dict(
                game=game,
                display_time=50,
                use_minimal_action_set=True,
                sticky_action_prob=0,
                difficulty_ramping=False
            ),
        )


def get_fn(init, final, ftype='linear'):
    if init == final:
        return init
    else:
        if ftype == 'linear':
            return get_linear_fn(init, final, 1.)
        elif ftype == 'exp':
            def f(p):
                return init * np.exp(np.log(final / init) * (1 - p))
            return f
        elif ftype == 'inverse':
            def f(p):
                k = (init / final) - 1
                return init / (1 + k * (1 - p))
            return f


class VecTranspose(VecEnvWrapper):
    """
    Re-order channels, from HxWxC to CxHxW.
    It is required for PyTorch convolution layers.

    :param venv:
    """

    def __init__(self, venv: VecEnv):

        observation_space = self.transpose_space(venv.observation_space)
        super(VecTranspose, self).__init__(venv, observation_space=observation_space)

    @staticmethod
    def transpose_space(observation_space: spaces.Box, key: str = "") -> spaces.Box:
        """
        Transpose an observation space (re-order channels).

        :param observation_space:
        :param key: In case of dictionary space, the key of the observation space.
        :return:
        """
        # Sanity checks
        height, width, channels = observation_space.shape
        new_shape = (channels, height, width)
        return spaces.Box(low=0, high=255, shape=new_shape, dtype=observation_space.dtype)

    @staticmethod
    def transpose_image(image: np.ndarray) -> np.ndarray:
        """
        Transpose an image or batch of images (re-order channels).

        :param image:
        :return:
        """
        if len(image.shape) == 3:
            return np.transpose(image, (2, 0, 1))
        return np.transpose(image, (0, 3, 1, 2))

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()

        # Transpose the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.transpose_image(infos[idx]["terminal_observation"])

        return self.transpose_image(observations), rewards, dones, infos

    def reset(self):
        """
        Reset all environments
        """
        return self.transpose_image(self.venv.reset())

    def close(self) -> None:
        self.venv.close()


class VecLogger(VecEnvWrapper):
    """
    A vectorized logger

    :param venv: The vectorized environment
    :param filename: the location to save a log file, can be None for no log
    """

    def __init__(
        self,
        venv: VecEnv,
        logdir: Optional[str] = None,
    ):
        # Avoid circular import
        from stable_baselines3.common.monitor import ResultsWriter

        VecEnvWrapper.__init__(self, venv)
        self.t_start = time.time()

        if logdir:
            os.makedirs(logdir, exist_ok=True)
            filename = os.path.join(logdir, '0')
            self.results_writer = ResultsWriter(filename, header={"t_start": self.t_start})
        else:
            self.results_writer = None

        self.scores = []
        self.steps = 0

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        for done, info in zip(dones, infos):
            if done and 'episode' in info:
                ep_info = info['episode']
                self.scores.append(ep_info['r'])
                self.steps += ep_info['l']
                if self.results_writer:
                    episode_info = {"r": ep_info['r'], "l": ep_info['l'], "t": round(time.time() - self.t_start, 6)}
                    self.results_writer.write_row(episode_info)
        return obs, rewards, dones, infos

    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return self.venv.close()
