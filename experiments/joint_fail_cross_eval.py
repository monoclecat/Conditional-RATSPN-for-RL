import os
import re
from typing import List, Dict
import yaml
import platform
from dataclasses import dataclass
import wandb
import numpy as np
import torch as th
from stable_baselines3.common.evaluation import evaluate_policy

from joint_fail_sac_sb3 import RunConfig, create_joint_fail_env, EntropyLoggingSAC


@dataclass
class JointFailProbEvalConfig(RunConfig):
    model_path: str
    num_eval_ep: int


def find_exp_dirs(parent_dir: str) -> List[str]:
    if not os.path.isdir(parent_dir):
        return []
    dir_contents = os.listdir(parent_dir)
    if 'wandb' in dir_contents:
        return [parent_dir]
    else:
        list_of_lists = [find_exp_dirs(os.path.join(parent_dir, c)) for c in dir_contents]
        flattened_list = [d for d_list in list_of_lists for d in d_list]
        return flattened_list


def find_wandb_run_dir_in_exp_dir(exp_dir: str):
    wandb_run_dir = None
    if os.path.exists(wandb_path := os.path.join(exp_dir, 'wandb')):
        wandb_run_dir = [fn for fn in os.listdir(wandb_path) if fn.startswith("run")]
    if wandb_run_dir is None:
        return wandb_run_dir
    wandb_run_dir.sort(key=lambda x: os.path.getctime(os.path.join(wandb_path, x)))
    wandb_run_dir = wandb_run_dir[0]
    return wandb_run_dir


def without_intermediate_value_keys(d):
    if not isinstance(d, dict):
        return d
    if len(d.keys()) == 2 and 'value' in d.keys():
        return without_intermediate_value_keys(d['value'])
    return {k: without_intermediate_value_keys(v) for k, v in d.items()}


def eval_over_joint_fail_probs(config: JointFailProbEvalConfig, train_config: Dict = None):
    if train_config is None:
        train_config = {}

    if config.no_video:
        os.environ['LD_PRELOAD'] = os.environ.get('CONDA_PREFIX') + '/lib/libGLEW.so'

    if config.log_dir:
        assert os.path.exists(config.log_dir), f"The log_dir doesn't exist! {config.log_dir}"

    wandb.login(key=os.environ['WANDB_API_KEY'])

    run_name = os.path.split(config.log_dir)[1]
    monitor_path = os.path.join(config.log_dir, "joint_fail_prob_monitor")
    for d in [config.log_dir, monitor_path]:
        os.makedirs(d, exist_ok=True)

    seed_regex = re.compile('_*s(|eed)[0-9]+')
    run_group = None
    if (match := seed_regex.search(run_name)) is not None:
        cut_out = match.span()
        run_group = run_name[:cut_out[0]] + run_name[cut_out[1]:]
    wandb_run = wandb.init(
        dir=config.log_dir,
        project=config.proj_name,
        name=run_name,
        group=run_group,
        sync_tensorboard=True,
        monitor_gym=True,
        reinit=True,
        force=True,
    )
    config.run_id = wandb_run.id
    wandb_run.config.update({
        **vars(config),
        **train_config,
        'machine': platform.node(),
    })
    config.to_yaml(os.path.join(config.log_dir, 'config.yaml'))

    jf_prob_key = 'joint_fail_prob'
    ep_rew_prefix = 'ep_rew'
    ep_len_prefix = 'ep_len'
    wandb.define_metric(jf_prob_key)
    wandb.define_metric(ep_rew_prefix + '/*', step_metric=jf_prob_key)
    wandb.define_metric(ep_rew_prefix + '/*', step_metric=jf_prob_key)

    for joint_fail_prob in np.arange(0, 1.01, 0.1):
        env = create_joint_fail_env(
            joint_fail_prob=joint_fail_prob, sample_failing_joints=config.sample_failing_joints,
            sample_failures_every=config.sample_failures_every,
            env_name=config.env_name, num_envs=config.num_envs, no_video=config.no_video, log_dir=config.log_dir,
            save_interval=config.save_interval,
        )

        model = EntropyLoggingSAC.load(config.model_path, env)

        np.random.seed(config.seed)
        th.manual_seed(config.seed)

        ep_rews, ep_lens = evaluate_policy(model=model, env=env, n_eval_episodes=config.num_eval_ep,
                                           return_episode_rewards=True)
        log_dict = {
            jf_prob_key: joint_fail_prob,
            ep_rew_prefix + '/min': np.min(ep_rews),
            ep_rew_prefix + '/mean': np.mean(ep_rews),
            ep_rew_prefix + '/max': np.max(ep_rews),
            ep_len_prefix + '/min': np.min(ep_lens),
            ep_len_prefix + '/mean': np.mean(ep_lens),
            ep_len_prefix + '/max': np.max(ep_lens),
        }
        wandb.log(log_dict)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', '-d', type=str, default='.',
                        help='Root directory containing joint_fail_sac_experiments.')
    parser.add_argument('--wandb_proj', '-p', type=str, default='test_joint_fail_cross_eval',
                        help='Project name for wandb')
    parser.add_argument('--num_eval_ep', '-n', type=int, default=10,
                        help='Number of episodes to evaluate model on per joint fail prob.')
    args = parser.parse_args()
    assert os.path.exists(args.root_dir)
    exp_dirs = find_exp_dirs(args.root_dir)
    step_pattern = re.compile('_(?P<step>[1-9]+0+)_steps')
    for exp in exp_dirs:
        log_prefix = f'{exp}\n\t'
        model_dir = os.path.join(exp, 'models')
        max_step_model = None
        max_step = 0
        for model in os.listdir(model_dir):
            if (m := step_pattern.search(model)) is not None:
                if (step := int(m.group('step'))) > max_step:
                    max_step = step
                    max_step_model = model
        if max_step_model is None:
            print(log_prefix + f"No max_step_model found in {model_dir}!")
            continue
        model_path = os.path.join(model_dir, max_step_model)

        wandb_run_dir = find_wandb_run_dir_in_exp_dir(exp)
        if wandb_run_dir is None:
            print(log_prefix + f"Couldn't find a wandb_run_dir")
            continue
        with open(os.path.join(exp, 'wandb', wandb_run_dir, 'files', 'config.yaml'), 'r') as file:
            experiment_config = yaml.safe_load(file)
            experiment_config = without_intermediate_value_keys(experiment_config)

        for k in list(experiment_config.keys()):
            experiment_config['train_' + k] = experiment_config.pop(k)

        config = JointFailProbEvalConfig(
            log_dir=exp, proj_name=args.wandb_proj, model_path=model_path, num_eval_ep=args.num_eval_ep,
            env_name='Ant-v3', num_envs=1, no_video=True, sample_failing_joints=True,
            sample_failures_every='step', save_interval=-1, seed=10,
        )
        eval_over_joint_fail_probs(config, train_config=experiment_config)
