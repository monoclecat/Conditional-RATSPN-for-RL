import os
import re
import yaml
import numpy as np

from joint_fail_sac_sb3 import find_exp_dirs, find_wandb_run_dir_in_exp_dir, without_intermediate_value_keys, \
    JointFailProbEvalConfig, eval_over_joint_fail_probs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Path to model to evaluate')
    args = parser.parse_args()
    assert os.path.exists(args.model_path)
    log_dir = os.path.split(os.path.split(args.model_path)[0])[0]

    config = JointFailProbEvalConfig(
        log_dir=log_dir, proj_name='none', model_path=args.model_path, num_eval_ep=10,
        env_name='Ant-v3', num_envs=1, no_video=False, sample_failing_joints=False,
        sample_failures_every='episode', save_interval=1000, seed=10,
        joint_fail_probs=[0.1], no_wandb=True,
    )
    eval_over_joint_fail_probs(config, train_config=None)
