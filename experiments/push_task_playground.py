import numpy as np
import cv2
import time
import collections
import pymunk
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from envs.pushenv import PushEnv


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', '-ag', type=int, default=4)
    args = parser.parse_args()

    steps = 2000

    # Initialize environments.
    seed = 0
    env = PushEnv(num_agents=args.num_agents)

    def vis(obs):
        plt.imshow(obs)
        plt.show()

    # Reset environments.
    obs = env.reset(seed=seed)
    vis(obs)

    max_rew = 0
    for step in range(1, steps + 1):
        if False:
            # center
            act = np.broadcast_to(np.expand_dims(env.screen_center, 0), env.action_shape)
        elif False:
            # top right
            act = np.broadcast_to(np.expand_dims(env.screen_size * np.asarray([1, 0]), 0), env.action_shape)
        else:
            # bottom left right
            act = np.broadcast_to(np.expand_dims(env.screen_size * np.asarray([0, 1]), 0), env.action_shape)
        obs, reward, done, info = env.step(act)
        print(reward)
        vis(obs)
        max_rew = max(max_rew, reward)
