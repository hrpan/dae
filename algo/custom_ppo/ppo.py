from typing import Any, Dict, Optional, Type, Union
from functools import partial

import numpy as np
import torch as th
import warnings

from gym import spaces
from algo.custom_ppo.policy import CustomActorCriticPolicy
from algo.custom_ppo.buffer import CustomBuffer

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate
from stable_baselines3.common.vec_env import VecEnv


class CustomPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) with direct advantage estimation

    Paper: https://arxiv.org/abs/1707.06347
    Code: Code borrowed from Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param learning_rate_vf: The learning rate for the value function, it can be a function
        of the current progress remaining (from 1 to 0) (only used when actor/critic are separeted)
    :param n_steps: The number of steps to run for each environment per update
    :param batch_size: Minibatch size
    :param batch_size_vf: Minibatch size for critic training (only used when actor/critic are separeted)
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param n_epochs_vf: Number of epoch when optimizing the surrogate loss for critic training (only used when actor/critic are separeted)
    :param gamma: Discount factor
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param ent_coef: Entropy coefficient for the loss calculation
    :param kl_coef: KL penalty coefficient for actor training
    :param vf_coef: Value function coefficient for the loss calculation
    :param shared: Use shared network for actor/critic (seperate training is)
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param advantage_normalization: normalize the estimated advantage before computing PPO loss
    :param full_action: update policy using all actions instead of just the sampled ones
    :param dae_correction: compute critic loss with DAE-style (multistep advantage) or DuelingDQN-style (first step advantage)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Type[CustomActorCriticPolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        learning_rate_vf: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 64,
        batch_size_vf: Optional[int] = 8,
        n_epochs: int = 4,
        n_epochs_vf: int = 4,
        gamma: float = 0.99,
        clip_range: Union[float, Schedule] = 0.2,
        ent_coef: float = 0.01,
        kl_coef: float = 0.0,
        vf_coef: float = 0.5,
        shared: bool = False,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None,
        advantage_normalization: bool = False,
        full_action: bool = True,
        dae_correction: bool = True,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(CustomPPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=0,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Discrete,),
        )

        self.batch_size = batch_size
        self.batch_size_vf = batch_size_vf
        self.learning_rate_vf = learning_rate_vf
        self.n_epochs = n_epochs
        self.n_epochs_vf = n_epochs_vf
        self.advantage_normalization = advantage_normalization
        self.full_action = full_action
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.shared = shared

        if not shared:
            warnings.warn(
                "Training with seperate actor/critic is deprecated, use at your own risk"
            )
        self.dae_correction = dae_correction

        self.discount_matrix = th.tensor(
            [
                [0 if j < i else self.gamma ** (j - i) for j in range(n_steps)]
                for i in range(n_steps)
            ],
            dtype=th.float32,
            device=self.device,
        )
        self.discount_vector = gamma ** th.arange(
            n_steps, 0, -1, dtype=th.float32, device=self.device
        )

        if _init_setup_model:
            self._setup_model()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: CustomBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.eval()

        rollout_buffer.reset()

        callback.on_rollout_start()

        for _ in range(n_rollout_steps):
            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs, device=self.device)
                actions, policies, log_policies, values = self.policy.forward(
                    obs_tensor
                )
            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)

            actions = actions.reshape(-1, 1)
            rollout_buffer.add(
                self._last_obs, actions, rewards, dones, policies, log_policies, values
            )

            self._last_obs = new_obs

        with th.no_grad():
            obs_tensor = th.as_tensor(new_obs, device=self.device)
            values, _ = self.policy.predict_value(obs_tensor)
        rollout_buffer.finalize(last_values=values)

        callback.on_rollout_end()

        return True

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.rollout_buffer = CustomBuffer(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
        )

        self.policy = self.policy_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            lr_schedule_vf=self.lr_schedule_vf,
            shared_features_extractor=self.shared,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)

    def _setup_lr_schedule(self) -> None:
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        self.lr_schedule_vf = get_schedule_fn(self.learning_rate_vf)

    def _update_learning_rate(self, optimizer, schedule, suffix=""):

        new_lr = schedule(self._current_progress_remaining)

        self.logger.record(f"train/learning_rate{suffix}", new_lr)

        update_learning_rate(optimizer, new_lr)

    def _normalize_advantage(self, advantages, policies, eps=1e-5):

        std = (policies * advantages.pow(2)).sum(dim=1).mean().sqrt()
        return advantages / (std + eps)

    def _value_loss(self, deltas, values, lasts):
        loss = th.cat(
            [
                (
                    self.discount_matrix[: len(d), : len(d)].matmul(d)
                    + l * self.discount_vector[-len(d) :]
                    - v
                ).square()
                for d, v, l in zip(deltas, values, lasts)
            ]
        ).mean()

        return loss

    def _policy_loss(
        self, advantages, log_policy, old_log_policy, actions, clip_range=None
    ):

        if self.full_action:

            ratio = th.exp(log_policy - old_log_policy)
            loss = -(advantages * th.exp(log_policy)).sum(dim=1).mean()
        else:
            adv = advantages.gather(-1, actions).flatten()
            logp = log_policy.gather(-1, actions).flatten()
            old_logp = old_log_policy.gather(-1, actions).flatten()
            ratio = th.exp(logp - old_logp)

            policy_loss_1 = adv * ratio
            policy_loss_2 = adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            loss = -th.min(policy_loss_1, policy_loss_2).mean()

        return loss, ratio

    def _train_shared(self) -> None:

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer, self.lr_schedule)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses, clip_fractions, gnorms = [], [], []
        losses, value_losses, pg_losses, kl_divs = [], [], [], []
        gnorm_max, gnorm_min = 0, float("inf")

        for epoch in range(self.n_epochs):

            for data in self.rollout_buffer.get_trajs(batch_size=self.batch_size):
                old_policies = data.old_policies
                old_log_policies = data.old_log_policies
                actions = data.actions
                rewards = data.rewards
                last_values = data.last_values
                lengths = data.lengths

                (
                    values,
                    advantages,
                    policies,
                    log_policies,
                    entropy,
                ) = self.policy.evaluate_state(data.observations, old_policies)

                # value loss
                values = values.flatten().split(lengths)
                if self.dae_correction:
                    deltas = (
                        rewards - advantages.gather(dim=1, index=actions).flatten()
                    ).split(lengths)
                    value_loss = self._value_loss(deltas, values, last_values)
                else:
                    advs = (
                        advantages.gather(dim=1, index=actions).flatten().split(lengths)
                    )
                    value_loss = th.cat(
                        [
                            (
                                self.discount_matrix[: len(a), : len(a)].matmul(r)
                                + l * self.discount_vector[-len(a) :]
                                - a
                                - v
                            ).square()
                            for r, a, v, l in zip(
                                rewards.split(lengths), advs, values, last_values
                            )
                        ]
                    ).mean()

                # kl divergence
                kl_loss = (
                    (old_policies * (old_log_policies - log_policies)).sum(dim=1).mean()
                )

                # normalize adv
                advantages = advantages.detach().clone()
                if self.advantage_normalization:
                    advantages = self._normalize_advantage(advantages, old_policies)

                # policy loss
                policy_loss, ratio = self._policy_loss(
                    advantages, log_policies, old_log_policies, actions, clip_range
                )

                # entropy loss
                entropy_loss = -th.mean(entropy)

                # full loss
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.kl_coef * kl_loss
                    + self.vf_coef * value_loss
                )

                losses.append(loss.item())
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = th.norm(
                    th.stack(
                        [
                            th.norm(p.grad)
                            for p in self.policy.parameters()
                            if p.grad is not None
                        ]
                    )
                ).item()
                gnorm_max = max(gnorm_max, gnorm)
                gnorm_min = min(gnorm_min, gnorm)

                # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.policy.optimizer.step()
                # Logging
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                kl_divs.append(kl_loss.item())
                gnorms.append(gnorm)

                self._n_updates += 1

        # Logs
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record(
            "train/policy_min", self.rollout_buffer.policies.min().item()
        )
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/gnorm", np.mean(gnorms))
        self.logger.record("train/gnorm_max", np.mean(gnorm_max))
        self.logger.record("train/gnorm_min", np.mean(gnorm_min))
        self.logger.record("train/approx_kl", np.mean(kl_divs))
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def _train_separate(self) -> None:

        # Update optimizer learning rate
        self._update_learning_rate(
            self.policy.optimizer, self.lr_schedule, suffix="_pi"
        )
        self._update_learning_rate(
            self.policy.optimizer_vf, self.lr_schedule_vf, suffix="_vf"
        )

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses, clip_fractions, kl_divs = [], [], []
        value_losses, pg_losses = [], []
        gnorm_pi, gnorm_vf = [], []

        # train for n_epochs epochs

        self.policy.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.n_epochs_vf):
            for data in self.rollout_buffer.get_trajs(batch_size=self.batch_size_vf):
                old_policies = data.old_policies
                old_log_policies = data.old_log_policies
                actions = data.actions
                rewards = data.rewards
                last_values = data.last_values
                lengths = data.lengths

                values, advantages = self.policy.predict_value(
                    data.observations, old_policies
                )
                # value loss
                values = values.flatten().split(lengths)
                deltas = (
                    rewards - advantages.gather(dim=1, index=actions).flatten()
                ).split(lengths)
                value_loss = self._value_loss(deltas, values, last_values)

                self.policy.optimizer_vf.zero_grad(set_to_none=True)
                value_loss.backward()
                gnorm = th.norm(
                    th.stack(
                        [
                            th.norm(p.grad)
                            for p in self.policy.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                self.policy.optimizer_vf.step()

                # logging
                value_losses.append(value_loss.item())
                gnorm_vf.append(gnorm.item())

        self.rollout_buffer.update_advantage(self.policy)
        self.policy.optimizer_vf.zero_grad(set_to_none=True)

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                old_policies = rollout_data.old_policies
                old_log_policies = rollout_data.old_log_policies
                actions = rollout_data.actions.long()
                advantages = rollout_data.advantages

                policies, log_policies, entropy = self.policy.predict_policy(
                    rollout_data.observations
                )

                # kl divergence
                kl_div = (
                    (old_policies * (old_log_policies - log_policies)).sum(dim=1).mean()
                )

                # Normalize advantage
                if self.advantage_normalization:
                    advantages = self._normalize_advantage(advantages, old_policies)

                # policy loss
                policy_loss, ratio = self._policy_loss(
                    advantages, log_policies, old_log_policies, actions, clip_range
                )

                # entropy loss
                entropy_loss = -th.mean(entropy)

                # full loss
                loss = (
                    policy_loss + self.ent_coef * entropy_loss + self.kl_coef * kl_div
                )

                # Optimization step
                self.policy.optimizer.zero_grad(set_to_none=True)

                loss.backward()
                gnorm = th.norm(
                    th.stack(
                        [
                            th.norm(p.grad)
                            for p in self.policy.parameters()
                            if p.grad is not None
                        ]
                    )
                ).item()

                # Clip grad norm
                # th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # Logging
                gnorm_pi.append(gnorm)
                pg_losses.append(policy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )
                entropy_losses.append(entropy_loss.item())
                kl_divs.append(kl_div.item())

        self._n_updates += self.n_epochs

        # Logs
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/gnorm_vf", np.mean(gnorm_vf))
        self.logger.record("train/gnorm_pi", np.mean(gnorm_pi))
        self.logger.record("train/approx_kl", np.mean(kl_divs))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        self.policy.train()

        if self.shared:
            return self._train_shared()
        else:
            return self._train_separate()

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "CustomPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "CustomPPO":

        return super(CustomPPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
