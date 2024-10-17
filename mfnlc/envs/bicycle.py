import random
from typing import Dict

import gym
import matplotlib.pyplot as plt
import numpy as np

from mfnlc.envs.base import EnvBase


class Bicycle(EnvBase):
    
    C_H = -37.1967
    C_M = 0.0342
    C_A = 1.9569
    L_F = 0.225
    L_R = 0.225

    K_P_THETA = 1.0
    K_P_V = 1.0
    
    STATE_SIZE = 4
    ACTION_SIZE = 2

    def __init__(self,
                 no_obstacle=False,
                 end_on_collision=False,
                 fixed_init_and_goal=False):
        super(Bicycle, self).__init__(no_obstacle,
                                              end_on_collision,
                                              fixed_init_and_goal)
        self.arrive_radius = 0.2
        self.robot_radius = 0.15
        self.obstacle_num = 20
        self.obstacle_in_obs = 2
        self.obstacle_radius = 0.15
        self.collision_penalty = -0.01
        self.arrive_reward = 20
        self.step_size = 0.1
        self.robot_name = "Bicycle"

        self.goal_size = 500
        self.subgoal_size = 100

        self.floor_lb = np.array([-10., -10.], dtype=np.float32)
        self.floor_ub = np.array([10., 10.], dtype=np.float32)

        self.init = None
        self.goal = None
        self.robot_pos = None
        self.obstacle_centers = None
        self.prev_subgoal_num = 0

        self.action_max = 5 # 5 m/s
        self._build_space()
        self._build_sample_space()
        self.prev_vec_to_goal = None

        self.fig, self.ax = None, None
        self.robot_patch = None
        self.roa_patch = None
        
        self.internal_state = None
        self.previous_goal_dist = None

    @property
    def hazards_pos(self):
        return self.obstacle_centers

    def _build_space(self):
        action_high = np.ones(self.ACTION_SIZE, dtype=np.float32) * self.action_max
        action_low = -action_high
        self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)

        if self.no_obstacle:
            observation_high = 2 * self.floor_ub[0] * np.ones(self.STATE_SIZE, dtype=np.float32)
        else:
            observation_high = 2 * np.ones(self.state_size + self.obstacle_in_obs * 2, dtype=np.float32)
            observation_high = (observation_high.reshape([-1, self.STATE_SIZE]) * self.floor_ub[0]).flatten()
        observation_low = -observation_high
        self.observation_space = gym.spaces.Box(observation_low, observation_high, dtype=np.float32)

    def _build_sample_space(self):
        self.position_list = []

        x_lb, y_lb = self.floor_lb
        x_ub, y_ub = self.floor_ub
        if self.no_obstacle:
            # no obstacle env is only used for training. Use small map during training.
            grid_num_per_line = int(0.3 * (x_ub - x_lb) / (self.robot_radius * 4))
            x = np.linspace(0.3 * x_lb, 0.3 * x_ub, num=grid_num_per_line, dtype=np.float32)
            y = np.linspace(0.3 * y_lb, 0.3 * y_ub, num=grid_num_per_line, dtype=np.float32)
        else:
            grid_num_per_line = int((x_ub - x_lb) / (self.robot_radius * 4))
            x = np.linspace(0.95 * x_lb, 0.95 * x_ub, num=grid_num_per_line, dtype=np.float32)
            y = np.linspace(0.95 * y_lb, 0.95 * y_ub, num=grid_num_per_line, dtype=np.float32)

        xv, yv = np.meshgrid(x, y)
        for i in range(len(x)):
            for j in range(len(y)):
                self.position_list.append([xv[i, j], yv[i, j]])

    def _generate_map(self):
        if not self.fixed_init_and_goal:
            if self.no_obstacle:
                positions = random.sample(self.position_list, 2)
            else:
                positions = random.sample(self.position_list, 2 + self.obstacle_num)
                self.obstacle_centers = np.array(positions[2:])

            self.init = np.array(positions[0], dtype=np.float32)
            self.goal = np.array(positions[1], dtype=np.float32)
        else:
            if self.goal is None or self.init is None:
                positions = random.sample(self.position_list, 2)
                self.init = np.array([0.0, 0.0], dtype=np.float32)
                self.goal = np.array([-1.0, 0.0], dtype=np.float32)

            if not self.no_obstacle:
                positions = random.sample(self.position_list, self.obstacle_num)

                # Lazy implementation. This may make the obstacle numbers be smaller
                if self.init.tolist() in positions:
                    positions.remove(self.init.tolist())  # noqa
                if self.goal.tolist() in positions:
                    positions.remove(self.goal.tolist())  # noqa
                self.obstacle_centers = np.array(positions)

    def update_env_config(self, config: Dict):
        self.__dict__.update(config)
        self._build_sample_space()
        self._build_space()
        # self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        super(Bicycle, self).reset()

        self._generate_map()
        self.robot_pos = self.init
        
        self.internal_state = np.zeros(4)
        self.internal_state[:2] = self.robot_pos

        self.previous_goal_dist = None
        self.prev_vec_to_goal = None
        self.prev_subgoal_num = 0

        plt.close("all")
        if self.fig is not None:
            self.fig, self.ax = None, None

        return self.get_obs()

    def goal_obs(self) -> np.ndarray:
        if self.subgoal is not None:
            goal_obs = self.subgoal - self.robot_pos
        else:
            goal_obs = self.goal - self.robot_pos
        return goal_obs

    def robot_obs(self) -> np.ndarray:
        return self.internal_state[2:] 
    
    def obstacle_obs(self) -> np.ndarray:
        if not self.no_obstacle:
            vec_to_obs = self.obstacle_centers - self.robot_pos
            dist_to_obs = np.linalg.norm(vec_to_obs, ord=2, axis=-1)
            order = dist_to_obs.argsort()[:self.obstacle_in_obs]

            return vec_to_obs[order].flatten()
        else:
            return np.array([])

    def collision_detection(self):
        if self.no_obstacle:
            return False

        closest_dist = np.min(np.linalg.norm(
            self.obstacle_centers - self.robot_pos, axis=-1, ord=2))
        return closest_dist < self.robot_radius + self.obstacle_radius

    def arrive(self):
        return np.linalg.norm(self.goal - self.robot_pos, ord=2) < self.arrive_radius
    
    def get_goal_reward(self):
        goal_dist = np.linalg.norm(self.goal_obs(), ord=2)
        if self.previous_goal_dist is None:
            goal_reward = 0.0
        else:
            goal_reward = (self.previous_goal_dist - goal_dist) * 10
        self.previous_goal_dist = goal_dist

        return goal_reward
    
    def step_internal_state(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ctrl = self.velocity_controller(action, self.internal_state)
        der = self.get_derivative(self.internal_state, ctrl)
        self.internal_state += der * self.step_size
        self.internal_state[:2] = self.internal_state[:2].clip(self.floor_lb, self.floor_ub)
        self.internal_state[3] = self.normalize_angle(self.internal_state[3])

    def step(self, action: np.ndarray):
        self.step_internal_state(action)
        self.robot_pos = self.internal_state[:2]

        collision = self.collision_detection()
        arrive = self.arrive()

        if self.end_on_collision and collision:
            done = True
        else:
            done = arrive

        reward = self.get_goal_reward() + collision * self.collision_penalty + arrive * self.arrive_reward

        self.traj.append(self.robot_pos)

        return self.get_obs(), reward, done, {"collision": collision, "goal_met": arrive}

    def render(self, mode="human"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))

            if not self.no_obstacle:
                for obstacle_center in self.obstacle_centers:
                    obstacle_patch = plt.Circle(
                        obstacle_center, radius=self.obstacle_radius, color="blue", alpha=0.5)
                    self.ax.add_patch(obstacle_patch)

            self.robot_patch = plt.Circle(
                self.robot_pos, radius=self.robot_radius, color="red", alpha=0.5)  # noqa
            self.ax.add_patch(self.robot_patch)

            self.roa_patch = plt.Circle(
                self.robot_pos, radius=self.robot_radius, color="cyan", alpha=0.5)
            self.ax.add_patch(self.roa_patch)

            self.ax.scatter(*self.goal, s=self.goal_size, marker='o', color='green', alpha=0.5)

            self.ax.set_xlim(self.floor_lb[0], self.floor_ub[0])
            self.ax.set_ylim(self.floor_lb[1], self.floor_ub[1])
            plt.axis('off')

        if len(self.subgoal_list) != self.prev_subgoal_num:
            self.ax.scatter(*self.subgoal_list[-1], s=self.subgoal_size, marker='o', color='green', alpha=0.5)
            self.prev_subgoal_num = len(self.subgoal_list)

        if len(self.traj) % self.render_config["traj_sample_freq"] == 0 and len(self.traj) > 0:
            self.ax.scatter(*self.traj[-1], s=self.subgoal_size / 3, marker='o', color='gold', alpha=0.5)

        self.robot_patch.center = self.robot_pos

        if self.roa_center is not None:
            self.roa_patch.center = self.roa_center
            self.roa_patch.radius = self.roa_radius

        self.fig.canvas.draw()

        if mode == "human":
            plt.pause(0.001)
        elif mode == "rgb_array":
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

            return data
    
    @classmethod
    def normalize_angle(cls, angle: float) -> float:
        normalized = angle
        while normalized > np.pi:
            normalized -= 2.0 * np.pi
        while normalized < -np.pi:
            normalized += 2.0 * np.pi
        return normalized

    @classmethod
    def velocity_controller(cls, v_des: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Simple velocity controller. It will make the robot move towards the desired velocity.
        :param v_des: desired velocity
        :param state: current state of the robot
        :return: heading and speed action to be applied to the robot
        """
        assert len(v_des) == cls.ACTION_SIZE
        assert len(state) == cls.STATE_SIZE
        
        cur_vx = state[2] * np.cos(state[3])
        cur_vy = state[2] * np.sin(state[3])
        
        e_vx = v_des[0] - cur_vx
        e_vy = v_des[1] - cur_vy
        
        theta_des = np.arctan2(v_des[1], v_des[0])
        e_theta = cls.normalize_angle(theta_des - state[3])
        
        e_longitudinal = max(e_vx * np.cos(state[3]) + e_vy * np.sin(state[3]), 0.1)
        throttle_input = (cls.K_P_V * e_longitudinal + cls.C_A * state[2]) / (cls.C_A * cls.C_M) + cls.C_H
        # penalize large heading errors
        if abs(e_theta) > np.pi / 2.0:
            throttle_input *= 1.0 - abs(e_theta) / np.pi
        
        heading_input = np.clip(cls.K_P_THETA * e_theta, -np.pi / 4.0, np.pi / 4.0)
        
        return np.array([heading_input, throttle_input])
    
    @classmethod
    def get_derivative(cls, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the state given the current state and action
        :param state: current state of the robot
        :param action: action to be applied to the robot
        :return: derivative of the state
        """
        assert len(state) == cls.STATE_SIZE
        assert len(action) == cls.ACTION_SIZE
        
        der = np.zeros(cls.STATE_SIZE)
        
        u = action[1]
        delta = action[0]
        v = state[2]
        theta = state[3]
        
        # x' = v * cos(theta)
        der[0] = v * np.cos(theta)
        
        # y' = v * sin(theta)
        der[1] = v * np.sin(theta)
        
        # v' = -ca * v + ca * cm * (u - ch)
        der[2] = -cls.C_A * v + cls.C_A * cls.C_M * (u - cls.C_H)
        
        # theta' = v * (1.0 / (lf + lr)) * tan(delta)
        der[3] = v * (1.0 / (cls.L_F + cls.L_R)) * np.tan(delta)
        
        return der


if __name__ == '__main__':
    env = Bicycle(no_obstacle=True, fixed_init_and_goal=True)
    obs = env.reset()
    for _ in range(50):
        action = obs[:2]
        obs, reward, done, info = env.step(action)
        print(obs)
        # print(reward)
    env.close()

