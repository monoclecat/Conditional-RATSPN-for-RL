import stable_baselines3 as sb3
import gym
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, BaseModel
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Any, Callable, Dict, Optional, Type, Union, Iterable

from cspn import CspnConfig, CSPN
from distributions import RatNormal
from stable_baselines3.common.utils import zip_strict


def polyak_update(
        params: Iterable[th.nn.Parameter],
        target_params: Iterable[th.nn.Parameter],
        tau: float,
) -> None:
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            if param.dtype != th.long:
                target_param.data.mul_(1 - tau)
                th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


class CspnActor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param Critic activation_fn: Inner activation function of the critic
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            features_extractor: nn.Module,
            features_dim: int,
            R: int,
            D: int,
            I: int,
            S: int,
            dropout: float,
            feat_layers: int,
            sum_param_layers: int,
            dist_param_layers: int,
            normalize_images: bool = True,
            cond_layers_inner_act: Type[nn.Module] = nn.ReLU,
            vi_ent_approx_sample_size: int = 5,
            squash_output: bool = True,
            min_std: float = 0.001,
            max_std: float = 1.0,
            **kwargs
    ):
        super(CspnActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=squash_output,
        )

        # Save arguments to re-create object at loading
        self.features_dim = features_dim
        self.vi_ent_approx_sample_size = vi_ent_approx_sample_size

        action_dim = get_action_dim(self.action_space)

        config = CspnConfig()
        config.F_cond = (features_dim,)
        config.C = 1
        config.feat_layers = feat_layers
        config.sum_param_layers = sum_param_layers
        config.dist_param_layers = dist_param_layers
        config.F = action_dim
        config.R = R
        config.D = D if D is not None else int(np.log2(action_dim))
        config.I = I
        config.S = S
        config.dropout = dropout
        config.leaf_base_class = RatNormal
        config.tanh_squash = squash_output
        config.cond_layers_inner_act = cond_layers_inner_act
        config.leaf_base_kwargs = {
            'min_sigma': min_std,
            'max_sigma': max_std,
        }
        self.config = config
        self.cspn = CSPN(config)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                mlp_inner_act=self.mlp_inner_act,
                log_std_init=self.log_std_init,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        features = self.extract_features(obs)
        if th.is_grad_enabled():
            action = self.cspn.sample_onehot_style(condition=features, is_mpe=deterministic).sample
        else:
            action = self.cspn.sample_index_style(condition=features, is_mpe=deterministic).sample
        return action.squeeze(0)

    def action_entropy(self, obs: th.Tensor, log_ent_metrics: bool = True) -> Tuple[th.Tensor, th.Tensor, dict]:
        # return action and entropy
        features = self.extract_features(obs)
        action = self.cspn.sample_onehot_style(condition=features, is_mpe=False).sample.squeeze(0).squeeze(0)
        entropy, vi_ent_log = self.cspn.vi_entropy_approx(
            condition=None, verbose=log_ent_metrics,
            sample_size=self.vi_ent_approx_sample_size,
        )
        return action, entropy, vi_ent_log

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class CspnPolicy(SACPolicy):
    """
    Policy class (with a CSPN actor and an MLP critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param critic_arch: The specification of the value networks.
    :param mlp_inner_act: Activation function
    :param log_std_init: Initial value for the log standard deviation
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            actor_cspn_args: dict,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            critic_activation_fn: Type[nn.Module] = nn.ReLU,
            log_std_init: float = -3,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            squash_output: bool = True,
            **kwargs
    ):
        super(SACPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [256, 256]

        if isinstance(net_arch, dict) and 'pi' in net_arch.keys():
            warnings.warn("CspnPolicy ignores pi net_arch settings, as the Cspn needs different configuration.")

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.actor_kwargs.update(actor_cspn_args)

        self.critic_kwargs = self.net_args.copy()
        _, critic_arch = get_actor_critic_arch(net_arch)
        self.critic_arch = critic_arch
        self.critic_activation_fn = critic_activation_fn

        self.critic_kwargs.update(
            {
                "activation_fn": self.critic_activation_fn,
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CspnActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs['squash_output'] = self.squash_output
        return CspnActor(**actor_kwargs).to(self.device)


class CspnSAC(SAC):
    """
    Soft Actor-Critic (SAC) modified for a CSPN actor
    """
    def __init__(self, **kwargs):
        super(CspnSAC, self).__init__(**kwargs)
        if self.action_space.bounded_above.all() and ~self.action_space.bounded_above.any() or \
                not self.action_space.bounded_above.all() and self.action_space.bounded_above.any():
            raise NotImplementedError("Case not covered yet where only part of the action space is bounded")

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, critic_values, = [], [], []
        actor_ent_log = []
        vi_ent_log = None

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, entropy, vi_ent_log = self.actor.action_entropy(replay_data.observations)
            entropy = entropy.reshape(-1, 1)
            actor_ent_log.append(entropy.mean().item())

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (self.target_entropy - entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_entropy, _ = self.actor.action_entropy(replay_data.next_observations,
                                                                          log_ent_metrics=False)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values + ent_coef * next_entropy.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            q_values = th.cat(current_q_values, dim=1)
            q_values, _ = th.min(q_values, dim=1, keepdim=True)
            critic_values.append(q_values.mean().item())

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(entropy - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = -(ent_coef * entropy + min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            # for lay in [*self.actor.cspn.feat_layers, *self.actor.cspn.dist_layers, *self.actor.cspn.sum_layers]:
            #     if isinstance(lay, th.nn.Linear) and lay.weight.grad.isnan().any():
            #         print(2)
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            if False:
                samples = self.actor.cspn.sample_index_style(
                    condition=replay_data.observations.float(), n=10, is_mpe=False, start_at_layer=1)
                nr_nodes, n, w, d, r = samples.shape
                first_rep = samples[:, :, :, :, 0]
                actions = first_rep.view(nr_nodes * n * w, d)
                expanded_obs = replay_data.observations.unsqueeze(0).unsqueeze(0).expand(nr_nodes, n, -1, -1)
                expanded_obs = expanded_obs.reshape(nr_nodes * n * w, expanded_obs.size(-1))
                q_vals = th.cat(self.critic(expanded_obs, actions), dim=1)
                min_q_vals, _ =  th.min(q_vals, dim=1, keepdim=True)
                min_q_vals = min_q_vals.view(nr_nodes, n, w)
                one_state = min_q_vals[:, :, 0]
                print(1)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_entropy", np.mean(actor_ent_log))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/critic_values", np.mean(critic_values))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if vi_ent_log is not None:
            for layer_id, layer_dict in vi_ent_log.items():
                for key, val in layer_dict.items():
                    self.logger.record(f"vi_ent_approx/sum_layer{layer_id}/{key}", np.mean(val))

    def _sample_action(
            self,
            learning_starts: int,
            action_noise: Optional[ActionNoise] = None,
            n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box) and self.actor.squash_output:
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case or action space unbounded, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action


class EntropyLoggingSAC(SAC):
    """
    Soft Actor-Critic (SAC) where the entropy is logged.
    """

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        log_prob = None

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/entropy", log_prob.detach().mean().item())
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
