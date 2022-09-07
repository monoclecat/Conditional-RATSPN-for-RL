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
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            num_agents,
            max_episode_len,
            agent_spawn_max_wall_dist,
            object_spawn_min_wall_dist,
    ):
        super(PushEnv, self).__init__()
        self._side_len = 512
        assert 0 < agent_spawn_max_wall_dist < self._side_len / 2
        self.agent_spawn_max_wall_dist = agent_spawn_max_wall_dist
        assert 0 < object_spawn_min_wall_dist < self._side_len / 2
        self.object_spawn_min_wall_dist = object_spawn_min_wall_dist

        self._steps = None
        self.__max_steps = max_episode_len
        self.success_threshold = None
        self.max_score = None
        self.goal_pose = None
        self.goal_color = None
        self.block = None
        self.cache_video = None
        self.teleop = None
        self.agents = None
        self.space = None

        self._num_agents = num_agents
        self._screen_size = (self._side_len, self._side_len)
        self.screen_center = self._screen_size[0] * 0.5, self._screen_size[1] * 0.5

        self._action_shape = (self._num_agents * 2,)
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
        self.control_hz = 100

        # Start PyMunk for physics.
        self.draw_options = DrawOptions(self.screen, agent_color=self._agent_color)
        self.sim_hz = 100

        # Local controller params.
        self.k_p, self.k_v = 100, 20  # PD control.

    @property
    def max_steps(self):
        return self.__max_steps

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

    def get_screen_img(self, in_observation_shape: bool = True):
        # pygame.display.flip()
        image = np.uint8(pygame.surfarray.array3d(self.screen).transpose((1, 0, 2)))
        if in_observation_shape:
            image = cv2.resize(image, dsize=self._observation_shape[:2])
        return image

    def observation(self):
        # Get observation.
        self.clear_screen()
        self.draw_objects(agents=False, objects=True)
        no_agent_image = self.get_screen_img(in_observation_shape=True)
        self.draw_objects(agents=True, objects=False)
        image = self.get_screen_img(in_observation_shape=True)
        obs = {
            'no_agent_image': no_agent_image,
            'image': image,
            'pos_agent': [a.position for a in self.agents],
            'vel_agent': [a.velocity for a in self.agents],
            'block_pose': (self.block.position, self.block.angle),
            'goal_pose': self.goal_pose
        }
        diff_angle = (obs['block_pose'][1] - obs['goal_pose'][1]) % np.pi
        diff_angle = min(diff_angle, np.pi - diff_angle)
        obs['diff_angle'] = diff_angle
        return obs

    def reset(self):
        self._steps = 0
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

        # Add agents, block, and goal zone.
        self.agents = []
        agent_radius = 15
        for _ in range(self._num_agents):
            wall = np.random.randint(0, 4)
            if wall == 0:
                # top wall
                spawn_x = np.random.randint(low=agent_radius+1, high=self.agent_spawn_max_wall_dist)
                spawn_y = np.random.randint(low=0, high=self._side_len)
            elif wall == 1:
                # left wall
                spawn_x = np.random.randint(low=0, high=self._side_len)
                spawn_y = np.random.randint(low=agent_radius + 1, high=self.agent_spawn_max_wall_dist)
            elif wall == 2:
                # bottom wall
                spawn_x = np.random.randint(low=agent_radius+1, high=self.agent_spawn_max_wall_dist)
                spawn_x = self._side_len - spawn_x
                spawn_y = np.random.randint(low=0, high=self._side_len)
            else:
                # right wall
                spawn_x = np.random.randint(low=0, high=self._side_len)
                spawn_y = np.random.randint(low=agent_radius + 1, high=self.agent_spawn_max_wall_dist)
                spawn_y = self._side_len - spawn_y

            self.agents.append(self.add_agent((spawn_x, spawn_y), agent_radius))

        self.block = self.add_tee((
            np.random.randint(low=self.object_spawn_min_wall_dist,
                              high=self._side_len - self.object_spawn_min_wall_dist),
            np.random.randint(low=self.object_spawn_min_wall_dist,
                              high=self._side_len - self.object_spawn_min_wall_dist),
        ), np.random.randn() * 2 * np.pi - np.pi)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = ((256, 256), np.pi / 4)  # x, y, theta (in radians)

        self.clear_screen()
        self.goal_area = self.num_pixels_with_goal_color(self.get_screen_img())
        self.max_score = 1000.0
        self.success_threshold = 0.95
        obs = self.observation()
        return self.mask_obs(obs)

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

    def clear_screen(self):
        """
        Clears screen and draws goal.
        """
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

    def draw_objects(self, agents: bool = True, objects: bool = True):
        # Draw agent and block.
        with self.draw_options.draw_ctx(draw_agents=agents, draw_objects=objects):
            self.space.debug_draw(self.draw_options)
        return self.screen

    def render(self, mode='human'):
        self.clear_screen()
        self.draw_objects()
        return self.get_screen_img(in_observation_shape=False)

    def step(self, act: np.ndarray):
        """
        Step the environment.

        Args:
            act: np.ndarray with shape (num_agents * 2,) of actions between -1.0 and +1.0.
                The actions are mapped to the agents like this:
                [agent0_x , agent0_y , agent1_x , agent1_y , ...]
                x and y in the range of (-1.0, +1.0) are mapped to the pixel coords (0, self._side_len)
                x == -1.0 is at the left of the env
                y == -1.0 is at the top of the env

        Returns: new observation, reward, done, info

        """
        # TODO clip actions either here or in agent
        assert act.shape == self._action_shape, f"Action must have shape {self._action_shape}"
        assert act.min() >= -1.0 and act.max() <= 1.0
        act = (act/2 + 0.5) * self._side_len
        act = [pymunk.Vec2d(act[i*2], act[i*2 + 1]) for i in range(self._num_agents)]
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
            # self.screen = self.render()
        obs = self.observation()

        # Get score (reward). Hide agents so their overlapping of the goal isn't counted
        visible_goal_pixels = self.num_pixels_with_goal_color(obs['no_agent_image'])
        reward, done = self.reward_from_visible_goal_area(visible_goal_pixels)

        # Done?
        self._steps += 1
        done = self._steps >= self.max_steps
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
