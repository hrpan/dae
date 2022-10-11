from typing import Optional, Union, List, NamedTuple, Generator
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from gym import spaces

import numpy as np
import torch as th


class CustomSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    old_policies: th.Tensor
    old_log_policies: th.Tensor
    advantages: th.Tensor


class CustomTrajSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    old_policies: th.Tensor
    old_log_policies: th.Tensor
    values: th.Tensor
    last_values: List
    lengths: List


class CustomBuffer(BaseBuffer):
    """
    Custom rollout buffer for PPO with DAE.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param horizon: Truncation length for MC backup
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):

        super(CustomBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        self._observations = np.zeros(
            (self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32
        )
        self._actions = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.long)
        self._rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._policies = np.zeros(
            (self.buffer_size, self.n_envs, self.action_space.n), dtype=np.float32
        )
        self._log_policies = np.zeros(
            (self.buffer_size, self.n_envs, self.action_space.n), dtype=np.float32
        )
        self._values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.observations = (
            self.actions
        ) = self.rewards = self.policies = self.log_policies = self.values = None
        self._observations.fill(0)
        self._actions.fill(0)
        self._rewards.fill(0)
        self._policies.fill(0)
        self._log_policies.fill(0)
        self.last_pos = [0] * self.n_envs
        self.trajectories = []
        self.lengths = []
        self.pos = 0
        super(CustomBuffer, self).reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        policy: th.Tensor,
        log_policy: th.Tensor,
        value: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param done: End of episode signal.
        :param policy: Policy from actor
        :param log_policy: log(Policy) from actor
        """

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self._observations[self.pos] = obs
        self._actions[self.pos] = action
        self._rewards[self.pos] = reward
        self._policies[self.pos] = policy.cpu().numpy()
        self._log_policies[self.pos] = log_policy.cpu().numpy()
        self._values[self.pos] = value.cpu().flatten().numpy()
        self.pos += 1
        for eid, d in enumerate(done):
            if d:
                last_p, p = self.last_pos[eid], self.pos
                self.trajectories.append(
                    (
                        self._observations[last_p:p, eid],
                        self._actions[last_p:p, eid],
                        self._rewards[last_p:p, eid],
                        self._policies[last_p:p, eid],
                        self._log_policies[last_p:p, eid],
                        self._values[last_p:p, eid],
                        0,
                    )
                )
                self.last_pos[eid] = self.pos
        if self.pos == self.buffer_size:
            self.full = True

    def finalize(self, last_values) -> None:

        for eid in range(self.n_envs):
            last_p = self.last_pos[eid]
            if last_p < self.buffer_size:
                self.trajectories.append(
                    (
                        self._observations[last_p:, eid],
                        self._actions[last_p:, eid],
                        self._rewards[last_p:, eid],
                        self._policies[last_p:, eid],
                        self._log_policies[last_p:, eid],
                        self._values[last_p:, eid],
                        last_values[eid, 0].item(),
                    )
                )

        _obs, _act, _rew, _pol, _lpol, _val, _last = zip(*self.trajectories)

        self.lengths = [len(o) for o in _obs]
        self.observations = th.as_tensor(np.concatenate(_obs), device=self.device)
        self.actions = th.as_tensor(np.concatenate(_act), device=self.device)
        self.rewards = th.as_tensor(np.concatenate(_rew), device=self.device)
        self.policies = th.as_tensor(np.concatenate(_pol), device=self.device)
        self.log_policies = th.as_tensor(np.concatenate(_lpol), device=self.device)
        self.values = th.as_tensor(np.concatenate(_val), device=self.device)
        self.last_values = list(_last)

        self.start_indices = np.insert(np.cumsum(self.lengths), 0, 0)[:-1]
        self.end_indices = np.cumsum(self.lengths)
        self.advantages = th.empty(
            self.policies.shape, dtype=th.float32, device=self.device
        )

    def update_value(self, policy, batch_size=1024):
        self.values = th.empty(
            (self.buffer_size * self.n_envs, 1), dtype=th.float32, device=self.device
        )
        start = 0
        size = len(self.observations)
        while start < size:
            end = min(start + batch_size, size)
            data = self._get_samples(th.arange(start, end, device=self.device))
            with th.no_grad():
                v, _ = policy.predict_value(data.observations, data.old_policies)
            self.values[start:end] = v
            start = end
        self.values = self.values.cpu()

    def update_advantage(self, policy, batch_size=1024):
        start = 0
        size = len(self.observations)
        while start < size:
            end = min(start + batch_size, size)
            _obs = self.observations[start:end]
            _pol = self.policies[start:end]
            with th.no_grad():
                _, adv = policy.predict_value(_obs, _pol)
            self.advantages[start:end] = adv
            start = end

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""

        indices = th.randperm(len(self.observations), device=self.device)

        if batch_size is None:
            batch_size = len(indices)

        start_idx = 0
        while start_idx < len(indices):
            yield self._get_samples(
                indices[start_idx : min(start_idx + batch_size, len(indices))]
            )
            start_idx += batch_size

    def _get_samples(self, indices) -> CustomSamples:

        data = (
            self.observations.index_select(dim=0, index=indices),
            self.actions.index_select(dim=0, index=indices),
            self.rewards.index_select(dim=0, index=indices),
            self.policies.index_select(dim=0, index=indices),
            self.log_policies.index_select(dim=0, index=indices),
            self.advantages.index_select(dim=0, index=indices),
        )

        return CustomSamples(*tuple(data))

    def get_trajs(self, batch_size: Optional[int] = None):
        assert self.full, ""

        if batch_size is None:
            batch_size = len(self.observations)

        indices = np.random.permutation(len(self.start_indices))

        start_idx = 0
        while start_idx < len(indices):
            traj_indices = []
            total_frames = 0
            while total_frames < batch_size and start_idx < len(indices):
                t_idx = indices[start_idx]
                traj_indices.append(t_idx)
                total_frames += self.end_indices[t_idx] - self.start_indices[t_idx]
                start_idx += 1
            if total_frames < batch_size:
                break
            yield self._get_traj_samples(traj_indices)

    def _get_traj_samples(self, indices) -> CustomTrajSamples:

        _obs, _act, _rew, _pol, _lpol, _val, _last, splits = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for idx in indices:
            start, end = self.start_indices[idx], self.end_indices[idx]
            _obs.append(self.observations[start:end])
            _act.append(self.actions[start:end])
            _rew.append(self.rewards[start:end])
            _pol.append(self.policies[start:end])
            _lpol.append(self.log_policies[start:end])
            _val.append(self.values[start:end])
            _last.append(self.last_values[idx])
            splits.append(end - start)

        return CustomTrajSamples(
            th.cat(_obs),
            th.cat(_act),
            th.cat(_rew),
            th.cat(_pol),
            th.cat(_lpol),
            th.cat(_val),
            _last,
            splits,
        )
