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

    steps = 20000

    # Initialize environments.
    seed = 0
    env = PushEnv(num_agents=args.num_agents)

    def vis(obs):
        plt.imshow(obs['image'])
        plt.show()

    # Reset environments.
    obs, reward, done, info = env.reset(seed=seed)
    vis(obs)

    score = 0
    for step in range(1, steps + 1):
        if False:
            act = np.random.rand(args.num_agents, 2) * 2 - 1
        else:

            act = np.zeros((args.num_agents, 2))
        obs, reward, done, info = env.step(act)
        score = max(score, obs['score'])
        vis(obs)
