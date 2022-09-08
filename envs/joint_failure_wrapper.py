import gym
import numpy as np

class JointFailureWrapper(gym.Wrapper):
    """
    A wrapper for gym environments that lets joint fail with a certain probability.
    It is possible to sample the action of a failed joint instead.
    Information about the joint failure is appended to the observation.
    """
    def __init__(self, env, joint_failure_prob: float, sample_failing_joints: bool):
        super().__init__(env)
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

    def _sample_joint_failures(self):
        failing_joints = (np.random.random_sample(self.action_space.shape) < 0.1).astype(self.observation_space.dtype)
        low = self.action_space.low
        high = self.action_space.high
        if self.sample_failing_joints:
            uniform_sample = low + np.random.random_sample(self.action_space.shape) * (high - low)
            uniform_sample = uniform_sample * failing_joints
        else:
            uniform_sample = np.zeros(self.action_space.shape)
        return np.hstack((failing_joints, uniform_sample))

    def step(self, action: np.ndarray):
        fails = self._sample_joint_failures()
        action_mask = -(fails[:self.action_space.shape[0]] - 1)
        action = action * action_mask
        samples = fails[self.action_space.shape[0]:]
        action = action + samples

        observation, reward, done, info = super(JointFailureWrapper, self).step(action)
        observation = np.hstack((observation, fails))
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = super(JointFailureWrapper, self).reset(**kwargs)
        return np.hstack((observation, np.zeros(self.action_space.shape), np.zeros(self.action_space.shape)))
