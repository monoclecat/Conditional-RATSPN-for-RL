import gym
import numpy as np
import sb3
from sb3 import CspnActor, CspnSAC
import os
import platform

import torch.nn as nn

from cspn import CSPN, print_cspn_params

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecVideoRecorder
from alr_envs_sb3_integration import make_vec_env

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2', help='Gym environment to train on.')
    parser.add_argument('--seed', '-s', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on. cpu or cuda.')
    parser.add_argument('--exp_name', type=str, default='test',
                        help='Experiment name. Will appear in name of saved model.')
    parser.add_argument('--save_dir', type=str, default='../../cspn_rl_experiments',
                        help='Directory to save the model to.')
    parser.add_argument('--model_paths', '-models', nargs='+', type=str,
                        help='Absolute path to the pretrained model.')
    parser.add_argument('--episode_len', type=int, default=200, help='Reset env after this many steps.')
    parser.add_argument('--video', action='store_true',
                        help="Record a video of the agent. Env will not render for you to see.")
    args = parser.parse_args()

    args.model_paths = [
        # Ant-v3 experiments
        '/home/temp_store/andrew_delay/log/test_alr_envs_mp/lr_3e-6_no_ent_loss_s10/models/model.zip',
    ]

    for model_path in args.model_paths:
        print(f"Loading {model_path}")
        if model_path:
            assert os.path.exists(model_path), f"The model_path doesn't exist! {model_path}"

        model_dir = os.path.join('/', *model_path.split("/")[:-1])
        model_name = model_path.split("/")[-1].split(".")[0]

        results_path = os.path.join(model_dir, "inspect")
        for d in [results_path]:
            if not os.path.exists(d):
                os.makedirs(d)

        env = make_vec_env(
            env_id=args.env,
            n_envs=1,
            seed=args.seed,
            # monitor_dir=results_path,
            # monitor_dir=os.path.join(results_path, f"log_{args.exp_name}.txt"),
            # vec_env_cls=SubprocVecEnv,
            # vec_env_kwargs={'start_method': 'fork'},
        )

        model = CspnSAC.load(model_path, env)

        print(model.actor)
        print(model.critic)
        if isinstance(model.actor, CspnActor):
            print_cspn_params(model.actor.cspn)
        else:
            print(f"Actor MLP has {sum(p.numel() for p in model.actor.parameters() if p.requires_grad)} parameters.")

        obs = env.reset()

        # Without env as a VecVideoRecorder we need the env var LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so;
        if args.video:
            env.metadata['video.frames_per_second'] = 5
            env.metadata['video.output_frames_per_second'] = 30
            env = VecVideoRecorder(env, results_path,
                                   record_video_trigger=lambda x: x == 0, video_length=args.episode_len,
                                   name_prefix=f"{model_name}")

        env.reset()
        if not args.video:
            env.render()
        step = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step += 1
            if not args.video:
                env.render()
            if done or step > args.episode_len:
                if args.video:
                    break
                obs = env.reset()
                step = 0
        env.close()
