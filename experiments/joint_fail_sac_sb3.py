import re
import time

import yaml
import numpy as np
import os
import platform
import wandb
from wandb.integration.sb3 import WandbCallback

import torch as th
import torch.nn as nn

from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import FlattenExtractor, NatureCNN

from cspn import CSPN, print_cspn_params
from sb3 import *
from utils import non_existing_folder_name
from envs.joint_failure_wrapper import wrap_in_float_and_joint_fail
from dataclasses import dataclass


@dataclass
class RunConfig:
    log_dir: str
    proj_name: str
    env_name: str
    num_envs: int
    no_video: bool
    sample_failures_every: str
    sample_failing_joints: bool
    save_interval: int
    seed: int

    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(vars(self), f)


@dataclass
class TrainConfig(RunConfig):
    buffer_size: int
    cspn_depth: int
    device: str
    dist_param_layers: list
    dropout: float
    ent_coef: float
    feat_layers: list
    joint_fail_prob: float
    learning_rate: float
    learning_starts: int
    log_interval: int
    mlp_actor: bool
    naive_sample_size: int
    no_wandb: bool
    num_dist: int
    num_sums: int
    objective: str
    recurs_sample_size: int
    repetitions: int
    sum_param_layers: list
    total_timesteps: int
    run_id: str = None
    stop_after_nr_of_save_intervals: int = None

    @staticmethod
    def from_yaml(path: str):
        assert os.path.exists(path)
        with open(path, 'r') as f:
            config: dict = yaml.safe_load(f)
        return TrainConfig(**config)


def get_max_step_from_sb3_model_checkpoints(model_path):
    step_expr = re.compile('.*[^0-9](?P<step>[0-9]+)_*steps')
    matches = [step_expr.match(p) for p in os.listdir(model_path)]
    steps = [int(m.group('step')) for m in matches if m is not None]
    if steps:
        return max(steps)
    else:
        return None


def create_joint_fail_env(
        joint_fail_prob: float, sample_failing_joints: bool, env_name: str, num_envs: int, no_video: bool,
        sample_failures_every: str, log_dir: str, save_interval: int,
):
    joint_fail_kwargs = {'joint_failure_prob': joint_fail_prob,
                         'sample_failing_joints': sample_failing_joints,
                         'sample_failures_every': sample_failures_every}
    env = make_vec_env(
        env_id=env_name,
        n_envs=num_envs,
        monitor_dir=os.path.join(log_dir, 'monitor'),
        wrapper_class=wrap_in_float_and_joint_fail,
        wrapper_kwargs=joint_fail_kwargs,
        # vec_env_cls=SubprocVecEnv,
        # vec_env_kwargs={'start_method': 'fork'},
    )

    if not no_video:
        # Without env as a VecVideoRecorder we need the env var LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so
        env = VecVideoRecorder(env, video_folder=os.path.join(log_dir, 'video'),
                               record_video_trigger=lambda x: x % save_interval == 0,
                               video_length=1000)
    return env


def train_joint_fail_sac(config: TrainConfig):
    assert os.environ.get('LD_PRELOAD') is None, "The LD_PRELOAD environment variable may not be set externally."
    if config.no_video:
        os.environ['LD_PRELOAD'] = os.environ.get('CONDA_PREFIX') + '/lib/libGLEW.so'

    if config.save_interval > config.total_timesteps:
        config.save_interval = config.total_timesteps

    if config.log_dir:
        assert os.path.exists(config.log_dir), f"The log_dir doesn't exist! {config.log_dir}"

    if not config.no_wandb:
        wandb.login(key=os.environ['WANDB_API_KEY'])

    run_name = os.path.split(config.log_dir)[1]
    monitor_path = os.path.join(config.log_dir, "monitor")
    model_path = os.path.join(config.log_dir, "models")
    video_path = os.path.join(config.log_dir, "video")
    for d in [config.log_dir, monitor_path, model_path, video_path]:
        os.makedirs(d, exist_ok=True)

    run = None
    if config.run_id is not None:
        run = wandb.init(
            id=config.run_id,
            resume='must',
            dir=config.log_dir,
            project=config.proj_name,
            sync_tensorboard=True,
            monitor_gym=True,
            reinit=True,
            force=True,
        )
    elif not config.no_wandb:
        seed_regex = re.compile('_*s(|eed)[0-9]+')
        run_group = None
        if (match := seed_regex.search(run_name)) is not None:
            cut_out = match.span()
            run_group = run_name[:cut_out[0]] + run_name[cut_out[1]:]
        run = wandb.init(
            dir=config.log_dir,
            project=config.proj_name,
            name=run_name,
            group=run_group,
            sync_tensorboard=True,
            monitor_gym=True,
            reinit=True,
            force=True,
        )
        config.run_id = run.id
        run.config.update({
            **vars(config),
            'machine': platform.node(),
        })

    config.to_yaml(os.path.join(config.log_dir, 'config.yaml'))

    env = create_joint_fail_env(
        joint_fail_prob=config.joint_fail_prob, sample_failing_joints=config.sample_failing_joints,
        sample_failures_every=config.sample_failures_every,
        env_name=config.env_name, num_envs=config.num_envs, no_video=config.no_video, log_dir=config.log_dir,
        save_interval=config.save_interval,
    )

    model_path = os.path.join(config.log_dir, 'models')
    if os.path.exists(model_path) and \
            (last_saved_step := get_max_step_from_sb3_model_checkpoints(model_path)) is not None:
        print(f"Continuing from step {last_saved_step}")
        files_of_last_saved_step = [p for p in os.listdir(model_path) if str(last_saved_step) in p]
        model_checkpoint = [p for p in files_of_last_saved_step if p.endswith('zip')]
        assert len(model_checkpoint) > 0
        model_checkpoint = os.path.join(model_path, model_checkpoint[0])
        model = EntropyLoggingSAC.load(model_checkpoint, env)
        if config.total_timesteps - model.num_timesteps <= 0:
            print("Model has already reached its total timesteps.")
            return
        replay_buffer = [p for p in files_of_last_saved_step if p.startswith('replay_buffer')]
        assert len(replay_buffer) > 0
        replay_buffer = os.path.join(model_path, replay_buffer[0])
        model.load_replay_buffer(replay_buffer)
    else:
        sac_kwargs = {
            'env': env,
            'seed': config.seed,
            'verbose': 2,
            'ent_coef': config.ent_coef,
            'learning_starts': config.learning_starts,
            'device': config.device,
            'learning_rate': config.learning_rate,
            'buffer_size': config.buffer_size,
        }
        cspn_args = {
            'R': config.repetitions,
            'D': config.cspn_depth,
            'I': config.num_dist,
            'S': config.num_sums,
            'dropout': config.dropout,
            'feat_layers': config.feat_layers,
            'sum_param_layers': config.sum_param_layers,
            'dist_param_layers': config.dist_param_layers,
            'cond_layers_inner_act': nn.LeakyReLU,  # nn.Identity if no_relu else nn.ReLU,
            'entropy_objective': config.objective,
            'recurs_ent_approx_sample_size': config.recurs_sample_size,
            'naive_ent_approx_sample_size': config.naive_sample_size,
        }
        sac_kwargs['policy_kwargs'] = {
            'actor_cspn_args': cspn_args,
            'joint_failure_info_in_obs': True,
            'features_extractor_class': NatureCNN if len(env.observation_space.shape) > 1 else FlattenExtractor,
        }
        if run is not None:
            run.config.update(sac_kwargs)
        model = EntropyLoggingSAC(policy=CustomMlpPolicy if config.mlp_actor else CspnPolicy, **sac_kwargs)

    train_model(
        model=model, seed=config.seed, total_timesteps=config.total_timesteps, log_dir=config.log_dir,
        save_interval_steps=config.save_interval, log_interval_episodes=config.log_interval, wandb_run=run,
        stop_after_nr_of_save_intervals=config.stop_after_nr_of_save_intervals,
    )


def train_model(
        model: EntropyLoggingSAC, seed: int, total_timesteps: int, log_dir: str, save_interval_steps: int,
        stop_after_nr_of_save_intervals: int = None, log_interval_episodes: int = 4, wandb_run=None,
):
    if total_timesteps - model.num_timesteps <= 0:
        print("Model has already reached its total timesteps.")
    else:
        np.random.seed(seed)
        th.manual_seed(seed)

        run_name = os.path.split(log_dir)[1]
        model_path = os.path.join(log_dir, "models")
        callback = [CheckpointCallbackSaveReplayBuffer(
            save_freq=save_interval_steps,
            save_path=model_path,
            name_prefix=run_name,
            stop_after_nr_of_saves=stop_after_nr_of_save_intervals,
        )]
        if wandb_run is not None:
            callback.append(WandbCallback(
                # gradient_save_freq=10000,
                # model_save_path=model_path,
                # model_save_freq=save_interval,
                verbose=2,
            ))

        logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        logger.output_formats[0].max_length = 50
        model.set_logger(logger)

        print(model.actor)
        print(model.critic)
        if isinstance(model.actor, CspnActor):
            print_cspn_params(model.actor.cspn)
        else:
            print(f"Actor MLP has {sum(p.numel() for p in model.actor.parameters() if p.requires_grad)} parameters.")
        # noinspection PyTypeChecker
        model.learn(
            total_timesteps=total_timesteps - model.num_timesteps,
            log_interval=log_interval_episodes,
            # Regarding reset_num_timesteps: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#id3
            # We set it to False so that the logging keeps the correct step information
            reset_num_timesteps=False,
            tb_log_name=run_name,
            callback=callback,
        )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        help='Path to an existing config. If provided, all other arguments will be ignored.')
    parser.add_argument('--seeds', '-s', type=int, nargs='+')
    parser.add_argument('--mlp_actor', action='store_true', help='Use a MLP actor')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of parallel environments to run.')
    parser.add_argument('--total_timesteps', '-ts', type=int, default=int(1e6), help='Total timesteps to train model.')
    parser.add_argument('--save_interval', type=int, help='Save model and a video every save_interval timesteps.')
    parser.add_argument('--stop_after_nr_of_save_intervals', type=int, default=None,
                        help='Stop early after a number of save_intervals. Disable by leaving this None. ')
    parser.add_argument('--log_interval', type=int, default=4, help='Log interval')
    parser.add_argument('--env_name', '-env', type=str, help='Gym environment to train on.')
    parser.add_argument('--device', '-dev', type=str, default='cuda', help='Device to run on. cpu or cuda.')
    parser.add_argument('--proj_name', '-proj', type=str, default='test_proj', help='Project name for WandB')
    parser.add_argument('--run_name', '-name', type=str, default='test_run',
                        help='Name of this run for WandB. The seed will be automatically appended. ')
    parser.add_argument('--log_dir', type=str, default='../../cspn_rl_experiments',
                        help='Directory to save logs to.')
    parser.add_argument('--no_wandb', action='store_true', help="Don't log this run in WandB")
    parser.add_argument('--no_video', action='store_true', help="Don't record videos of the agent.")
    # SAC arguments
    parser.add_argument('--ent_coef', type=float, default=0.1, help='Entropy temperature')
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--learning_starts', type=int, default=1000,
                        help='Nr. of steps to act randomly in the beginning.')
    parser.add_argument('--buffer_size', type=int, default=1_000_000, help='replay buffer size')
    parser.add_argument('--joint_fail_prob', '-jf', type=float, default=0.05, help="Joints can fail with this probability")
    parser.add_argument('--sample_failing_joints', action='store_true', help="Sample replacements for failing joints")
    parser.add_argument('--sample_failures_every', type=str, default='step', choices=['step', 'episode'],
                        help='When to sample joint failures.')
    # CSPN arguments
    parser.add_argument('--repetitions', '-R', type=int, default=3, help='Number of parallel CSPNs to learn at once. ')
    parser.add_argument('--cspn_depth', '-D', type=int,
                        help='Depth of the CSPN. If not provided, maximum will be used (ceil of log2(inputs)).')
    parser.add_argument('--num_dist', '-I', type=int, default=3, help='Number of Gauss dists per pixel.')
    parser.add_argument('--num_sums', '-S', type=int, default=3, help='Number of sums per RV in each sum layer.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout to apply')
    parser.add_argument('--feat_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN feature layers.')
    parser.add_argument('--sum_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN sum param layers.')
    parser.add_argument('--dist_param_layers', type=int, nargs='+',
                        help='List of sizes of the CSPN dist param layers.')
    parser.add_argument('--objective', '-obj', type=str, help='Entropy objective to maximize.', required=False)
    # Entropy arguments
    parser.add_argument('--recurs_sample_size', type=int, default=5,
                        help='Number of samples to approximate recursive entropy with. ')
    parser.add_argument('--naive_sample_size', type=int, default=50,
                        help='Number of samples to approximate naive entropy with. ')
    args = parser.parse_args()
    if args.config is not None:
        assert os.path.exists(args.config) and args.config.endswith('yaml')
        config = TrainConfig.from_yaml(args.config)
        train_joint_fail_sac(config)
    else:
        kwargs = vars(args)
        seeds = kwargs.pop('seeds')
        run_name = kwargs.pop('run_name')
        kwargs.pop('config')
        for seed in seeds:
            run_name = f"{'MLP' if args.mlp_actor else 'CSPN'}_{run_name}"
            run_name_seed = f"{run_name}_s{seed}"
            log_dir = os.path.join(args.log_dir, args.proj_name)
            log_dir = os.path.join(log_dir, non_existing_folder_name(log_dir, run_name_seed))
            kwargs['log_dir'] = log_dir
            train_joint_fail_sac(TrainConfig(seed=seed, **kwargs))
