# from IPython.display import Image

# @markdown ### **Environment**
# @markdown Defines a PyMunk-based environment (Johnny's T-pushing task).
# @markdown
# @markdown **Goal**: push the gray T-block into the green area.

import collections
import os
from typing import Optional

import cv2
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
from envs.pymunk_helpers import DrawOptions

os.environ['SDL_VIDEODRIVER'] = 'dummy'


class PushEnv():

    def __init__(self, num_agents):

        # Start PyGame for visualization.
        self.__screen_size = (512, 512)
        self.screen = pygame.display.set_mode((self.__screen_size[1], self.__screen_size[0]), flags=pygame.HIDDEN)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Roboto", 16)
        self.control_hz = 10
        self.__num_agents = num_agents

        # Start PyMunk for physics.
        self.draw_options = DrawOptions(self.screen)
        self.sim_hz = 100

        # Local controller params.
        self.k_p, self.k_v = 100, 20  # PD control.

    @property
    def num_agents(self):
        return self.__num_agents

    @property
    def screen_size(self):
        return self.__screen_size

    @property
    def screen_center(self):
        return self.__screen_size[0] * 0.5, self.__screen_size[1] * 0.5

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
        self.agents = [self.add_circle((np.random.randint(50, 450), np.random.randint(50, 450)), 15)
                       for _ in range(self.__num_agents)]
        # self.agent = self.agents[0]
        # self.agent = self.add_circle((np.random.randint(50, 450), np.random.randint(50, 450)), 15)
        self.block = self.add_tee((np.random.randint(100, 400), np.random.randint(100, 400)),
                                  np.random.randn() * 2 * np.pi - np.pi)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = ((256, 256), np.pi / 4)  # x, y, theta (in radians)

        self.max_score = 50 * 100
        self.success_threshold = 0.95  # 95% coverage.
        self.screen = self.render()
        return self.step()

    def render(self):

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

        # Draw agent and block.
        self.space.debug_draw(self.draw_options)

        # Info and flip screen.
        # self.screen.blit(self.font.render(f'FPS: {self.clock.get_fps():.1f}', True, pygame.Color('darkgrey')), (10, 10))
        # self.screen.blit(self.font.render('Push the gray block to the green target pose.', True, pygame.Color('darkgrey')), (10, self.__screen_size[0] - 35))
        # self.screen.blit(self.font.render('Press ESC or Q to quit.', True, pygame.Color('darkgrey')), (10, self.__screen_size[0] - 20))
        pygame.display.flip()
        self.cache_video.append(
            cv2.resize(np.uint8(pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)), dsize=(96, 96)))
        return self.screen

    def step(self, act: Optional[np.ndarray] = None):
        if act is not None:
            assert act.shape == (self.__num_agents, 2), f"Action must have shape (num_agents={self.__num_agents}, 2)"
            act = [pymunk.Vec2d(act[i, 0], act[i, 1]) for i in range(self.__num_agents)]
            dt = 1.0 / self.sim_hz
            for _ in range(self.sim_hz // self.control_hz):

                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)  # P control works too.
                for i in range(self.__num_agents):
                    agent = self.agents[i]
                    acceleration = self.k_p * (act[i] - agent.position) + self.k_v * (Vec2d(0, 0) - agent.velocity)
                    agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)
                if self.teleop:
                    self.clock.tick(self.sim_hz)  # Limit framerate.

                # Render screen.
                self.screen = self.render()

        # Get observation.
        image = np.uint8(pygame.surfarray.array3d(self.screen).transpose(1, 0, 2))
        obs = {'image': cv2.resize(image, dsize=(96, 96)),
               'pos_agent': self.agent.position,
               'vel_agent': self.agent.velocity,
               'block_pose': (self.block.position, self.block.angle),
               'goal_pose': self.goal_pose}
        diff_angle = (obs['block_pose'][1] - obs['goal_pose'][1]) % np.pi
        diff_angle = min(diff_angle, np.pi - diff_angle)
        obs['diff_angle'] = diff_angle

        # Get score (reward).
        score = (self.max_score - np.sum(
            np.mean(np.float32(image == np.uint8(self.goal_color)[:3]), axis=2) > 0)) / self.max_score
        obs['score'] = np.clip(score / self.success_threshold, 0, 1)

        # Done?
        done = obs['score'] > self.success_threshold
        for event in pygame.event.get():
            if (event.type == pygame.QUIT or event.type == pygame.KEYDOWN and (
                    event.key in [pygame.K_ESCAPE, pygame.K_q])):
                done = True
        return obs, 0, done, {}

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

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
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
