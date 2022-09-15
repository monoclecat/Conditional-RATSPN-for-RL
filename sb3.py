from abc import ABC

import stable_baselines3 as sb3
import gym
from stable_baselines3.sac.policies import SACPolicy, Actor
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
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
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

LOG_STD_MAX = 2
LOG_STD_MIN = -20

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


class JointFailureActor(BasePolicy):
    def __init__(
            self, *args, joint_failure_info_in_obs: bool = False, **kwargs):
        self.joint_failure_info_in_obs = joint_failure_info_in_obs
        super(JointFailureActor, self).__init__(*args, **kwargs)

    def extract_features(self, obs: th.Tensor) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        joint_failure_info = None
        if self.joint_failure_info_in_obs:
            obs, joint_failure_info = th.hsplit(obs, self.observation_space.shape)
            assert (isinstance(self, CustomMlpActor) and joint_failure_info.shape[1] == 0) or \
                   (isinstance(self, CspnActor) and joint_failure_info.shape[1] == self.action_space.shape[0] * 2)
        obs = super(JointFailureActor, self).extract_features(obs)
        return obs, joint_failure_info.to(dtype=obs.dtype)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        action, _, _ = self.action_entropy(
            obs=obs, deterministic=deterministic, log_ent_metrics=False, compute_entropy=False
        )
        return action

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class CspnActor(JointFailureActor):
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
            net_arch: List[int],
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
            entropy_objective: str,
            recurs_ent_approx_sample_size: int,
            naive_ent_approx_sample_size: int,
            cond_layers_inner_act: Type[nn.Module] = nn.ReLU,
            squash_output: bool = True,
            min_std: float = 0.001,
            max_std: float = 1.0,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            joint_failure_info_in_obs: bool = False,
            **kwargs
    ):
        super(CspnActor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            joint_failure_info_in_obs=joint_failure_info_in_obs,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.features_dim = features_dim
        self.recurs_ent_approx_sample_size = recurs_ent_approx_sample_size
        self.naive_ent_approx_sample_size = naive_ent_approx_sample_size
        self.entropy_objective = entropy_objective

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

    def _sample(self, condition: th.Tensor, is_mpe: bool, joint_failure_info: Optional[th.Tensor]):
        evidence = None
        if joint_failure_info is not None:
            joint_failure_info = joint_failure_info.to(dtype=self.cspn.dtype)
            failed_joints, evidence = th.hsplit(joint_failure_info, 2)
            evidence[failed_joints == 0.0] = th.nan
            evidence = evidence.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).atanh()
        action: th.Tensor = self.cspn.sample(
            mode='onehot' if th.is_grad_enabled() else 'index',
            condition=condition,
            is_mpe=is_mpe,
            evidence=evidence,  # evidence must be unsquashed, i.e. from inf support
        ).sample
        if joint_failure_info is not None:
            # The additional batch dimension of the evidence must be removed
            action = action.squeeze(0)
        # action shape [nr_nodes_sampled, nr_samples_of_each_node, nr_conditionals, nr_features, repetitions]
        # root node is sampled, with one sample per conditional, so shape is always [1, 1, w, action_shape, 1]
        action = action.squeeze(0).squeeze(0).squeeze(-1)
        return action

    def action_entropy(self, obs: th.Tensor, deterministic: bool = False, log_ent_metrics: bool = False,
                       compute_entropy: bool = False) -> Tuple[th.Tensor, th.Tensor, dict]:
        # return action and entropy
        features, joint_failure_info = self.extract_features(obs)
        action = self._sample(condition=features, is_mpe=deterministic, joint_failure_info=joint_failure_info)
        entropy = ent_log = None
        failed_joints, _ = th.hsplit(joint_failure_info, 2)
        if compute_entropy:
            if self.entropy_objective == 'recursive':
                entropy, ent_log = self.cspn.recursive_entropy_approx(
                    condition=features, verbose=log_ent_metrics,
                    sample_size=self.recurs_ent_approx_sample_size,
                    marginal_mask=failed_joints,
                )
            elif self.entropy_objective == 'naive':
                ent_log = {}
                entropy = self.cspn.naive_entropy_approx(
                    condition=features,
                    sample_size=self.naive_ent_approx_sample_size,
                    marginal_mask=failed_joints,
                )
            elif self.entropy_objective == 'huber':
                entropy, ent_log = self.cspn.huber_entropy_lb(
                    condition=features, verbose=log_ent_metrics,
                    marginal_mask=failed_joints,
                )
            else:
                raise ValueError(f"entropy_objective {self.entropy_objective} unknown!")
        return action, entropy, ent_log


class JointFailurePolicy(SACPolicy):
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
            self, *args,
            provide_joint_fail_info_to_critic: bool,
            joint_failure_prob: float,
            sample_failing_joints: bool,
            **kwargs
    ):
        self.joints_can_fail = joint_failure_prob > 0.0
        self.joint_failure_prob = joint_failure_prob
        self.sample_failing_joints = sample_failing_joints
        self.provide_joint_fail_info_to_critic = provide_joint_fail_info_to_critic
        super(JointFailurePolicy, self).__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        if self.provide_joint_fail_info_to_critic:
            self._include_joint_failure_info_in_obs_space(self.critic_kwargs)
        super(JointFailurePolicy, self)._build(lr_schedule)

    def _include_joint_failure_info_in_obs_space(self, ac_kwargs):
        orig_obs = ac_kwargs['observation_space']
        new_low = np.hstack((
            orig_obs.low,
            np.zeros_like(self.action_space.low),
            self.action_space.low
        ))
        new_high = np.hstack((
            orig_obs.high,
            np.ones_like(self.action_space.high),
            self.action_space.high
        ))
        ac_kwargs['observation_space'] = gym.spaces.Box(low=new_low, high=new_high, dtype=orig_obs.dtype)

    def sample_joint_failures(self, batch_size):
        action_shape = (batch_size,) + self.action_space.shape
        failing_joints = (np.random.random_sample(action_shape) < self.joint_failure_prob)
        low = self.action_space.low
        high = self.action_space.high
        if self.sample_failing_joints:
            uniform_sample = low + np.random.random_sample(action_shape) * (high - low)
            uniform_sample = uniform_sample * failing_joints
        else:
            uniform_sample = np.zeros(action_shape)
        fail_info = np.hstack((failing_joints, uniform_sample)).astype(self.observation_space.dtype)
        return th.as_tensor(fail_info, device=self.actor.device)

    def action_entropy_with_joint_failure(
            self,
            observation: th.Tensor,
            deterministic: bool = False,
            log_ent_metrics: bool = False,
            compute_entropy: bool = False
    ):
        joints_failed = self.sample_joint_failures(batch_size=observation.shape[0])
        observation = th.hstack((observation, joints_failed))
        unscaled_action, entropy, ent_log = self.actor.action_entropy(
            obs=observation,
            deterministic=deterministic,
            log_ent_metrics=log_ent_metrics,
            compute_entropy=compute_entropy,
        )

        failed_joints, samples = th.hsplit(joints_failed, 2)
        action_mask = -(failed_joints - 1)
        unscaled_action = unscaled_action * action_mask + samples
        return unscaled_action, entropy, ent_log

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        unscaled_action, entropy, ent_log = self.action_entropy_with_joint_failure(
            observation=observation, deterministic=deterministic, log_ent_metrics=False, compute_entropy=False
        )
        return unscaled_action

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        if self.provide_joint_fail_info_to_critic:
            assert isinstance(critic_kwargs['features_extractor'], FlattenExtractor)
            critic_kwargs['features_dim'] = critic_kwargs['observation_space'].low.shape[0]
        return ContinuousCritic(**critic_kwargs).to(self.device)


class CspnPolicy(JointFailurePolicy):
    def __init__(self, *args, actor_cspn_args: dict, **kwargs):
        self.actor_cspn_args = actor_cspn_args
        super(CspnPolicy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CspnActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs['squash_output'] = self.squash_output
        actor_kwargs['joint_failure_info_in_obs'] = self.joints_can_fail
        actor_kwargs.update(self.actor_cspn_args)
        return CspnActor(**actor_kwargs).to(self.device)


class CustomMlpActor(JointFailureActor, Actor):
    def __init__(
            self, *args, joint_failure_info_in_obs: bool = False, **kwargs
    ):
        self.joint_failure_info_in_obs = joint_failure_info_in_obs
        Actor.__init__(self, *args, **kwargs)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = th.hstack(self.extract_features(obs))
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def action_entropy(self, obs: th.Tensor, deterministic: bool = False, log_ent_metrics: bool = False,
                       compute_entropy: bool = False) -> Tuple[th.Tensor, th.Tensor, dict]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        if deterministic:
            actions_pi = self.action_dist.actions_from_params(
                mean_actions, log_std, deterministic=deterministic, **kwargs
            )
            return actions_pi, None, {}
        else:
            actions_pi, log_prob = self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
            return actions_pi, -log_prob, {}


class CustomMlpPolicy(JointFailurePolicy):
    def __init__(self, *args, actor_cspn_args=None, **kwargs):
        super(CustomMlpPolicy, self).__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        self._include_joint_failure_info_in_obs_space(self.actor_kwargs)
        super(JointFailurePolicy, self)._build(lr_schedule)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomMlpActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        assert isinstance(actor_kwargs['features_extractor'], FlattenExtractor)
        actor_kwargs['features_dim'] = actor_kwargs['observation_space'].low.shape[0]
        actor_kwargs['joint_failure_info_in_obs'] = self.joints_can_fail
        return CustomMlpActor(**actor_kwargs).to(self.device)


class CspnSAC(SAC):
    """
    Soft Actor-Critic (SAC) modified for a CSPN actor
    """
    def __init__(self, **kwargs):
        super(CspnSAC, self).__init__(**kwargs)
        if self.action_space.bounded_above.all() and ~self.action_space.bounded_above.any() or \
                not self.action_space.bounded_above.all() and self.action_space.bounded_above.any():
            raise NotImplementedError("Case not covered yet where only part of the action space is bounded")

    def learn(self, **kwargs) -> OffPolicyAlgorithm:
        if kwargs.get('tb_log_name') is None:
            kwargs['tb_log_name'] = 'CspnSAC'
        if isinstance(self.actor, CspnActor):
            if kwargs.get('callback') is None:
                kwargs['callback'] = CspnCallback()
            else:
                kwargs['callback'] = [CspnCallback(), kwargs['callback']]
        return super(CspnSAC, self).learn(**kwargs)

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

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            # actions_pi, entropy, _ = self.actor.action_entropy(replay_data.observations, log_ent_metrics=False)
            actions_pi, entropy, _ = self.policy.action_entropy_with_joint_failure(
                observation=replay_data.observations, deterministic=False, log_ent_metrics=False, compute_entropy=True
            )
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
                next_actions, next_entropy, _ = self.policy.action_entropy_with_joint_failure(
                    observation=replay_data.next_observations,
                    deterministic=False,
                    log_ent_metrics=False,
                    compute_entropy=True,
                )
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
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_entropy", np.mean(actor_ent_log))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/critic_values", np.mean(critic_values))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


class CspnCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CspnCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        self.model = None  # type: Optional[CspnSAC]
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        log_every = 10
        if self.model.num_timesteps > self.model.learning_starts \
                and (self.model.num_timesteps // self.model.n_envs) % log_every == 0:
            actor: CspnActor = self.model.actor
            cspn: CSPN = self.model.actor.cspn

            recursive_ent, recurs_log = cspn.recursive_entropy_approx(
                condition=None, verbose=True,
                sample_size=actor.recurs_ent_approx_sample_size,
            )
            naive_ent = cspn.naive_entropy_approx(
                condition=None,
                sample_size=actor.naive_ent_approx_sample_size,
            )
            huber_ent, huber_log = cspn.huber_entropy_lb(condition=None, verbose=True)
            log = {**recurs_log, **huber_log}
            log.update({
                'recursive_ent_approx': recursive_ent.detach().mean().item(),
                'huber_entropy_LB': huber_ent.detach().mean().item(),
                'naive_ent_approx': naive_ent.detach().mean().item(),
            })
            for key, val in log.items():
                self.model.logger.record(f"train/{key}", val)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class FloatWrapper(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super(FloatWrapper, self).__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low, high=self.observation_space.high, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=self.action_space.low, high=self.action_space.high, dtype=np.float32
        )


class EntropyLoggingSAC(SAC):
    """
    Soft Actor-Critic (SAC) for MLP policies where the entropy is logged.
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
