import gym
import numpy as np


def wrap_in_float_and_joint_fail(env, **kwargs):
    env = JointFailureWrapper(env, **kwargs)
    env = FloatWrapper(env)
    return env


class JointFailureWrapper(gym.Wrapper):
    """
    A wrapper for gym environments that lets joint fail with a certain probability.
    It is possible to sample the action of a failed joint instead.
    Information about the joint failure is appended to the observation.
    """
    def __init__(self, env, joint_failure_prob: float, sample_failing_joints: bool):
        super().__init__(env)
        self._next_joint_failure = self.no_joint_failures()
        self.joint_failure_prob = joint_failure_prob
        self.sample_failing_joints = sample_failing_joints
        self.orig_obs_space = self.observation_space
        new_low = np.hstack((
            self.orig_obs_space.low,
            np.zeros_like(self.action_space.low),
            self.action_space.low
        ))
        new_high = np.hstack((
            self.orig_obs_space.high,
            np.ones_like(self.action_space.high),
            self.action_space.high
        ))
        self.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=np.float64)

    def no_joint_failures(self):
        return np.hstack((np.zeros(self.action_space.shape), np.zeros(self.action_space.shape)))

    def sample_joint_failures(self):
        failing_joints = (np.random.random_sample(self.action_space.shape) < self.joint_failure_prob)
        failing_joints = failing_joints.astype(self.observation_space.dtype)
        low = self.action_space.low
        high = self.action_space.high
        if self.sample_failing_joints:
            uniform_sample = low + np.random.random_sample(self.action_space.shape) * (high - low)
            uniform_sample = uniform_sample * failing_joints
        else:
            uniform_sample = np.zeros(self.action_space.shape)
        self._next_joint_failure = np.hstack((failing_joints, uniform_sample))
        return self._next_joint_failure

    def step(self, action: np.ndarray):
        fails = self._next_joint_failure
        action_mask = -(fails[:self.action_space.shape[0]] - 1)
        action = action * action_mask
        samples = fails[self.action_space.shape[0]:]
        action = action + samples

        observation, reward, done, info = super(JointFailureWrapper, self).step(action)
        observation = np.hstack((observation, self.sample_joint_failures()))
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = super(JointFailureWrapper, self).reset(**kwargs)
        return np.hstack((observation, self.no_joint_failures()))


class FloatWrapper(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super(FloatWrapper, self).__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low, high=self.observation_space.high, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=self.action_space.low, high=self.action_space.high, dtype=np.float32
        )
