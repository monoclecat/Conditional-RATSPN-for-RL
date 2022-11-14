import os
import re
import yaml
import numpy as np

from joint_fail_sac_sb3 import find_exp_dirs, find_wandb_run_dir_in_exp_dir, without_intermediate_value_keys, \
    JointFailProbEvalConfig, eval_over_joint_fail_probs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', '-d', type=str, default='.',
                        help='Root directory containing joint_fail_sac_experiments.')
    parser.add_argument('--wandb_proj', '-p', type=str, default='test_joint_fail_cross_eval',
                        help='Project name for wandb')
    parser.add_argument('--num_eval_ep', '-n', type=int, default=10,
                        help='Number of episodes to evaluate model on per joint fail prob.')
    parser.add_argument('--min_fail_prob', type=float, default=0.0,
                        help='Minimum joint failure prob to evaluate')
    parser.add_argument('--max_fail_prob', type=float, default=0.5,
                        help='Maximum joint failure prob to evaluate')
    parser.add_argument('--fail_prob_steps', type=int, default=11,
                        help='Steps to evaluate between min and max.')
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
            joint_fail_probs=list(np.linspace(args.min_fail_prob, args.max_fail_prob, args.fail_prob_steps)),
            no_wandb=False,
        )
        eval_over_joint_fail_probs(config, train_config=experiment_config)
