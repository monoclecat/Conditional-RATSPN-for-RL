import numpy as np
import cv2
import time
import collections
import pymunk
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from envs.pushenv import PushEnv
from tqdm import tqdm
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', '-ag', type=int, default=4)
    args = parser.parse_args()

    steps = 100

    # Initialize environments.
    seed = 0
    env = PushEnv(
        num_agents=args.num_agents,
        max_episode_len=1,
        agent_spawn_max_wall_dist=30,
        object_spawn_min_wall_dist=100
    )

    def vis(obs):
        plt.imshow(obs)
        plt.show()

    # Reset environments.
    obs = env.reset(seed=seed)
    # check_env(env)
    vis(obs)

    max_rew = 0
    goal = -1
    for step in tqdm(range(1, steps + 1)):
        if goal == -1:
            act = np.random.random(env.action_shape) * 2 - 1
        elif goal == 0:
            # center
            act = np.zeros(env.action_shape)
        elif goal == 1:
            # top right
            act = np.tile(np.asarray([1, -1]), env.action_shape[0] // 2)
        elif goal == 2:
            # top left
            act = np.tile(np.asarray([-1, -1]), env.action_shape[0] // 2)
        elif goal == 3:
            # bottom left
            act = np.tile(np.asarray([-1, 1]), env.action_shape[0] // 2)
        elif goal == 4:
            # bottom right
            act = np.tile(np.asarray([1, 1]), env.action_shape[0] // 2)
        else:
            raise ValueError
        obs, reward, done, info = env.step(act)
        # print(reward)
        # vis(obs)
        if reward > max_rew:
            max_rew = reward
            print(f"New max reward {max_rew}")
        if done:
            env.reset()
    print(f"Max reward: {max_rew}")
