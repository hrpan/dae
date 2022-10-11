import torch as th
import torch.nn as nn
from functools import partial
from typing import Optional, Tuple, Type
import numpy as np
import gym
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN


class QVActorCriticPolicy(ActorCriticPolicy):

    """
    Policy class for indirect advantage estimation by estimating both Q(s,a) and V(s) simultaneously.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
    ):

        super(QVActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None,
            activation_fn=nn.Tanh,
            ortho_init=ortho_init,
            use_sde=False,
            log_std_init=0.0,
            full_std=True,
            sde_net_arch=None,
            use_expln=False,
            squash_output=False,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=None,
            normalize_images=True,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=None,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.
        """

        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.features_extractor.features_dim
        )

        self.value_net = nn.Linear(self.features_extractor.features_dim, 1)
        self.q_net = nn.Linear(
            self.features_extractor.features_dim, self.action_space.n
        )

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.q_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:

        latent = self.extract_features(obs)
        mean_actions = self.action_net(latent)
        distribution = self.action_dist.proba_distribution(mean_actions)

        return distribution.get_actions(deterministic=deterministic)

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        latent = self.extract_features(obs)
        mean_actions = self.action_net(latent)
        distribution = self.action_dist.proba_distribution(mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        policies = distribution.distribution.probs
        log_policies = distribution.distribution.logits
        values = self.value_net(latent)

        return actions, policies, log_policies, values

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        latent = self.extract_features(obs)

        distribution = self._get_action_dist_from_latent(latent)
        log_probs = distribution.log_prob(actions)

        q_values = self.q_net(latent)
        values = self.value_net(latent)
        return values, q_values, log_probs, distribution.entropy()

    def evaluate_state(
        self, obs: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        latent = self.extract_features(obs)
        distribution = self._get_action_dist_from_latent(latent)
        _policies = distribution.distribution.probs
        log_policies = distribution.distribution.logits

        q_values = self.q_net(latent)
        values = self.value_net(latent)
        return values, q_values, _policies, log_policies, distribution.entropy()

    def predict_policy(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        latent = self.extract_features(obs)

        distribution = self._get_action_dist_from_latent(latent)
        policies = distribution.distribution.probs
        log_policies = distribution.distribution.logits

        return policies, log_policies, distribution.entropy()

    def predict_value(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        latent = self.extract_features(obs)

        q_values = self.q_net(latent)
        values = self.value_net(latent)

        return values, q_values
