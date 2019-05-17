import os
import mujoco_py
import numpy as np
from gym.utils import seeding
import sys

from pprint import pprint


class JacoEnv():
    def __init__(self,
                 width,
                 height,
                 frame_skip,
                 rewarding_distance,
                 control_magnitude,
                 reward_continuous):
        self.frame_skip = frame_skip
        self.width = width
        self.height = height

        self.step_count = 0

        # Instantiate Mujoco model
        model_path = "jaco.xml"
        fullpath = os.path.join(
            os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model)

        self.init_state = self.sim.get_state()
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # Setup actuators
        self.actuator_bounds = self.sim.model.actuator_ctrlrange
        self.actuator_low = self.actuator_bounds[:, 0]
        self.actuator_high = self.actuator_bounds[:, 1]
        self.actuator_ctrlrange = self.actuator_high - self.actuator_low
        self.num_actuators = len(self.actuator_low)

        # init model_data_ctrl
        self.null_action = np.zeros(self.num_actuators)
        self.sim.data.ctrl[:] = self.null_action

        self.seed()

        self.sum_reward = 0
        self.rewarding_distance = rewarding_distance

        # Target position bounds
        self.target_bounds = np.array(((0.1, 0.5), (0.1, 0.5), (0.02, 0.02)))
        self.target_reset_distance = 0.2

        # Setup discrete action space
        self.control_values = self.actuator_ctrlrange * control_magnitude

        self.num_actions = 5
        self.action_space = np.asarray([list(range(self.num_actions))] * self.num_actuators)
        self.observation_space = ((0, ), (height, width, 3), (height, width, 3))

        # actions = [np.array([i, ] + [0 for j in range(self.num_actuators - 1)]) for i in range(5)]
        # for i in range(1, self.num_actuators):
        #     actions2 = []
        #     for _ in range(len(actions)):
        #         x = actions.pop(0)
        #         for j in range(5):
        #             cp = x.copy()
        #             cp[i] = j
        #             actions2.append(cp)
        #     actions = actions2

        actions = []
        for i in range(self.num_actuators):
            for j in range(self.num_actions):
                x = np.zeros(self.num_actuators)
                x[i] = j
                actions.append(x)
        self.action_dict = {k: v for k, v in enumerate(actions)}
        self.real_num_actions = len(self.action_dict)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_qpos_qvel(self, qpos, qvel):
        assert qpos.shape == (self.sim.model.nq, ) and qvel.shape == (
            self.sim.model.nv, )
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

    def reset(self):
        # Random initial position of Jaco
        # qpos = self.init_qpos + np.random.randn(self.sim.nv)
        self.step_count = 0

        #  Fixed initial position of Jaco
        qpos = self.init_qpos
        qvel = self.init_qvel

        # random object position start of episode
        self.reset_target()

        # set initial joint positions and velocities
        self.set_qpos_qvel(qpos, qvel)

        return self.get_obs()[2]

    def reset_target(self):
        # Randomize goal position within specified bounds
        self.goal = np.random.rand(3) * (self.target_bounds[:, 1] -
                                         self.target_bounds[:, 0]
                                         ) + self.target_bounds[:, 0]
        geom_positions = self.sim.model.geom_pos.copy()
        prev_goal_location = geom_positions[1]

        while (np.linalg.norm(prev_goal_location - self.goal) <
               self.target_reset_distance):
            self.goal = np.random.rand(3) * (self.target_bounds[:, 1] -
                                             self.target_bounds[:, 0]
                                             ) + self.target_bounds[:, 0]

        geom_positions[1] = self.goal
        self.sim.model.geom_pos[:] = geom_positions

    def render(self, camera_name=None, mode=None):
        rgb = self.sim.render(
            width=self.width, height=self.height, camera_name=camera_name)
        return rgb

    def _get_obs_joint(self):
        return np.concatenate(
            [self.sim.data.qpos.flat[:], self.sim.data.qvel.flat[:]])

    def _get_obs_rgb_view1(self):
        obs_rgb_view1 = self.render(camera_name='view1')
        return obs_rgb_view1

    def _get_obs_rgb_view2(self):
        obs_rgb_view2 = self.render(camera_name='view2')
        return obs_rgb_view2

    def get_obs(self):
        return (self._get_obs_joint(), self._get_obs_rgb_view1(),
                self._get_obs_rgb_view2())

    def do_simulation(self, ctrl):
        '''Do one step of simulation, taking new control as target

        Arguments:
            ctrl {np.array(num_actuator)}  -- new control to send to actuators
        '''
        ctrl = np.min((ctrl, self.actuator_high), axis=0)
        ctrl = np.max((ctrl, self.actuator_low), axis=0)

        self.sim.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            self.sim.step()

    def translate_action(self, action_int):
        return self.action_dict[action_int].copy()

    # @profile(immediate=True)
    def step(self, action_int):
        a = self.translate_action(action_int)
        dist = np.zeros(3)
        done = False
        if self.step_count >= 150:
            done = True
        new_control = np.copy(self.sim.data.ctrl).flatten()

        # Compute reward:
        # If any finger is close enough to target => +1
        dist[0] = np.linalg.norm(
            self.sim.data.get_body_xpos("jaco_link_finger_1") - self.goal)
        dist[1] = np.linalg.norm(
            self.sim.data.get_body_xpos("jaco_link_finger_2") - self.goal)
        dist[2] = np.linalg.norm(
            self.sim.data.get_body_xpos("jaco_link_finger_3") - self.goal)

        # if continuous reward
        reward = float(((np.mean(dist)+ 1e-6)**-1))

        if any(d < self.rewarding_distance for d in dist):
            # reward = 1
            done = True
            self.reset_target()

        # Transform discrete actions to continuous controls
        for i in range(self.num_actuators):
            '''
            0 = 0 velocity
            1 = small positive velocity
            2 = large positive velocity
            3 = small negative velocity
            4 = large negative velocity
            '''
            if a[i] == 0:
                new_control[i] = 0
            if a[i] == 1:
                new_control[i] = self.control_values[i] / 2
            if a[i] == 2:
                new_control[i] = self.control_values[i]
            if a[i] == 3:
                new_control[i] = -self.control_values[i] / 2
            elif a[i] == 4:
                new_control[i] = -self.control_values[i]

        # Do one step of simulation
        self.do_simulation(new_control)
        self.sum_reward += reward

        self.step_count += 1
        return self.get_obs()[2], reward, done, {}

    def change_floor_color(self, new_rgba):
        self.sim.model.geom_rgba[0] = new_rgba

    def change_cube_color(self, new_rgba):
        self.sim.model.geom_rgba[1] = new_rgba
