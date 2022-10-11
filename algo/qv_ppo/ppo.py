from typing import Any, Dict, Optional, Type, Union
from functools import partial

import numpy as np
import torch as th

from gym import spaces
from algo.qv_ppo.policy import QVActorCriticPolicy
from algo.custom_ppo.buffer import CustomBuffer

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate
from stable_baselines3.common.vec_env import VecEnv


class QVPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) with learned Q/V functions

    Code: Code borrowed from Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param ent_coef: Entropy coefficient for the loss calculation
    :param kl_coef: KL penalty coefficient for actor training
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
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
        policy: Type[QVActorCriticPolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 64,
        n_epochs: int = 4,
        gamma: float = 0.99,
        clip_range: Union[float, Schedule] = 0.2,
        ent_coef: float = 0.01,
        kl_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None,
        advantage_normalization: bool = False,
        full_action: bool = True,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(QVPPO, self).__init__(
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
        self.n_epochs = n_epochs
        self.advantage_normalization = advantage_normalization
        self.full_action = full_action
        self.clip_range = clip_range
        self.kl_coef = kl_coef

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
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)

    def _setup_lr_schedule(self) -> None:
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _update_learning_rate(self, optimizer, schedule, suffix=""):

        new_lr = schedule(self._current_progress_remaining)

        self.logger.record(f"train/learning_rate{suffix}", new_lr)

        update_learning_rate(optimizer, new_lr)

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

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        self.policy.train()

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
                    q_values,
                    policies,
                    log_policies,
                    entropy,
                ) = self.policy.evaluate_state(data.observations)

                # value loss
                values = values.flatten()
                q_values = q_values.gather(dim=1, index=actions).flatten()
                targets = th.cat(
                    [
                        self.discount_matrix[: len(r), : len(r)].matmul(r)
                        + l * self.discount_vector[-len(r) :]
                        for r, l in zip(rewards.split(lengths), last_values)
                    ]
                )
                v_loss = (values - targets).square().mean()
                q_loss = (q_values - targets).square().mean()
                value_loss = 0.5 * (v_loss + q_loss)

                # kl divergence
                kl_loss = (
                    (old_policies * (old_log_policies - log_policies)).sum(dim=1).mean()
                )

                # normalize adv
                advantages = (q_values - values).detach().clone()
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

                # policy loss

                logp = log_policies.gather(-1, actions).flatten()
                old_logp = old_log_policies.gather(-1, actions).flatten()
                ratio = th.exp(logp - old_logp)

                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

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

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "QVPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "QVPPO":

        return super(QVPPO, self).learn(
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
