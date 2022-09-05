# from IPython.display import Image

# @markdown ### **Environment**
# @markdown Defines a PyMunk-based environment (Johnny's T-pushing task).
# @markdown
# @markdown **Goal**: push the gray T-block into the green area.

import collections
import os
from typing import Optional

import gym
from gym import spaces
import cv2
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
from envs.pymunk_helpers import DrawOptions
import matplotlib.pyplot as plt

os.environ['SDL_VIDEODRIVER'] = 'dummy'


class PushEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents):
        super(PushEnv, self).__init__()

        # define state and action spaces
        self.success_threshold = None
        self.max_score = None
        self.goal_pose = None
        self.goal_color = None
        self.block = None
        self.cache_video = None
        self.teleop = None
        self.agents = None
        self.space = None

        self._side_len = 512
        self._num_agents = num_agents
        self._screen_size = (self._side_len, self._side_len)
        self.screen_center = self._screen_size[0] * 0.5, self._screen_size[1] * 0.5

        self._action_shape = (self._num_agents, 2)
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=self._action_shape)

        self._obs_mask = 'image'
        if self._obs_mask == 'image':
            self._observation_shape = (96, 96, 3)
            self._observation_space = spaces.Box(low=0, high=255, shape=self._observation_shape, dtype=np.uint8)
        else:
            raise NotImplemented("Currently, only the observation mask 'image' is supported. ")

        self._agent_color = pygame.Color('RoyalBlue')
        # Start PyGame for visualization.
        self.screen = pygame.display.set_mode(self._screen_size, flags=pygame.HIDDEN)
        self.clock = pygame.time.Clock()
        # self.font = pygame.font.SysFont("Roboto", 16)
        self.control_hz = 10

        # Start PyMunk for physics.
        self.draw_options = DrawOptions(self.screen, agent_color=self._agent_color)
        self.sim_hz = 100

        # Local controller params.
        self.k_p, self.k_v = 100, 20  # PD control.

    @property
    def action_shape(self):
        return self._action_shape

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_shape(self):
        return self._observation_shape

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def screen_size(self):
        return self._screen_size

    def mask_obs(self, obs):
        if self._obs_mask == 'image':
            return obs['image']
        else:
            raise NotImplemented("Currently, only the observation mask 'image' is supported. ")

    def get_screen_img(self, hide_agents=False, goal_only=False):
        self.screen = self.draw_space(hide_agents=hide_agents, goal_only=goal_only)
        no_agent_img = np.uint8(pygame.surfarray.array3d(self.screen).transpose((1, 0, 2)))
        return no_agent_img

    def observation(self):
        # Get observation.
        image = np.uint8(pygame.surfarray.array3d(self.screen).transpose((1, 0, 2)))
        obs = {
            'image': cv2.resize(image, dsize=self._observation_shape[:2]),
            'pos_agent': [a.position for a in self.agents],
            'vel_agent': [a.velocity for a in self.agents],
            'block_pose': (self.block.position, self.block.angle),
            'goal_pose': self.goal_pose
        }
        diff_angle = (obs['block_pose'][1] - obs['goal_pose'][1]) % np.pi
        diff_angle = min(diff_angle, np.pi - diff_angle)
        obs['diff_angle'] = diff_angle
        return obs

    def reset(self, seed=0):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.cache_video = []

        # Add walls.
        walls = [self.add_segment((5, 506), (5, 5), 2),
                 self.add_segment((5, 5), (506, 5), 2),
                 self.add_segment((506, 5), (506, 506), 2),
                 self.add_segment((5, 506), (506, 506), 2)]
        self.space.add(*walls)

        # Set random seed.
        np.random.seed(seed)

        # Add agents, block, and goal zone.
        self.agents = [self.add_agent((np.random.randint(50, 450), np.random.randint(50, 450)), 15)
                       for _ in range(self._num_agents)]
        # self.agent = self.agents[0]
        # self.agent = self.add_circle((np.random.randint(50, 450), np.random.randint(50, 450)), 15)
        self.block = self.add_tee((np.random.randint(100, 400), np.random.randint(100, 400)),
                                  np.random.randn() * 2 * np.pi - np.pi)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = ((256, 256), np.pi / 4)  # x, y, theta (in radians)

        self.goal_area = self.num_pixels_with_goal_color(self.get_screen_img(goal_only=True))
        self.max_score = 1000.0
        self.success_threshold = 0.95
        self.screen = self.render()
        return self.mask_obs(self.observation())

    def num_pixels_with_goal_color(self, img):
        goal_color_match = (np.sum(img == np.uint8(self.goal_color)[:3], axis=2) == 3)
        return goal_color_match.sum()

    def reward_from_visible_goal_area(self, visible_goal_pixels):
        diff = self.goal_area - visible_goal_pixels
        ratio = diff / self.goal_area
        assert 0.0 <= ratio <= 1.0
        reward = ratio * self.max_score
        done = bool(reward > self.max_score * self.success_threshold)
        return reward, done

    def draw_space(self, hide_agents: bool = False, goal_only: bool = False):
        # Clear screen.
        self.screen.fill(pygame.Color("white"))

        # Draw goal pose.
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position, body.angle = self.goal_pose
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(body.local_to_world(v), self.draw_options.surface) for v in
                           shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(self.draw_options.surface, self.goal_color, goal_points)

        assert not goal_only or not hide_agents
        if not goal_only:
            # Draw agent and block.
            with self.draw_options.draw_agent_ctx(hide=hide_agents):
                self.space.debug_draw(self.draw_options)
        return self.screen

    def render(self, mode='human'):
        self.draw_space()

        # Info and flip screen.
        # self.screen.blit(self.font.render(f'FPS: {self.clock.get_fps():.1f}', True, pygame.Color('darkgrey')), (10, 10))
        # self.screen.blit(self.font.render('Push the gray block to the green target pose.', True, pygame.Color('darkgrey')), (10, self._screen_size[0] - 35))
        # self.screen.blit(self.font.render('Press ESC or Q to quit.', True, pygame.Color('darkgrey')), (10, self._screen_size[0] - 20))
        pygame.display.flip()
        self.cache_video.append(
            cv2.resize(np.uint8(pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))),
                       dsize=self._observation_shape[:2]))
        return self.screen

    def step(self, act: np.ndarray = None):
        # TODO clip actions either here or in agent
        assert act.shape == (self._num_agents, 2), f"Action must have shape (num_agents={self._num_agents}, 2)"
        assert act.min() >= -1.0 and act.max() <= 1.0
        act = (act/2 + 0.5) * self._side_len
        act = [pymunk.Vec2d(act[i, 0], act[i, 1]) for i in range(self._num_agents)]
        dt = 1.0 / self.sim_hz
        for _ in range(self.sim_hz // self.control_hz):

            # Step PD control.
            # self.agent.velocity = self.k_p * (act - self.agent.position)  # P control works too.
            for i in range(self._num_agents):
                agent = self.agents[i]
                acceleration = self.k_p * (act[i] - agent.position) + self.k_v * (Vec2d(0, 0) - agent.velocity)
                agent.velocity += acceleration * dt

            # Step physics.
            self.space.step(dt)
            if self.teleop:
                self.clock.tick(self.sim_hz)  # Limit framerate.

            # Render screen.
            self.screen = self.render()
        obs = self.observation()

        # Get score (reward). Hide agents so their overlapping of the goal isn't counted
        visible_goal_pixels = self.num_pixels_with_goal_color(self.get_screen_img(hide_agents=True))
        reward, done = self.reward_from_visible_goal_area(visible_goal_pixels)

        # Done?
        for event in pygame.event.get():
            if (event.type == pygame.QUIT or event.type == pygame.KEYDOWN and (
                    event.key in [pygame.K_ESCAPE, pygame.K_q])):
                done = True
        return self.mask_obs(obs), reward, done, {}

    def close(self):
        super(PushEnv, self).close()

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return TeleopAgent(act)

    def add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')  # https://htmlcolorcodes.com/color-names
        return shape

    def add_agent(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = self._agent_color
        self.space.add(body, shape)
        return body

    def add_circle(self, position, radius):
        raise NotImplemented
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('LightGray')
        assert shape.color != self._agent_color
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length * scale / 2, scale),
                     (length * scale / 2, scale),
                     (length * scale / 2, 0),
                     (-length * scale / 2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale / 2, scale),
                     (-scale / 2, length * scale),
                     (scale / 2, length * scale),
                     (scale / 2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
