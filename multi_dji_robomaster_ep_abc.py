from abc import ABC, abstractmethod
import numpy as np
from qpsolvers import solve_qp
from math import cos, sin

class MultiDJIRoboMasterEP(ABC):
    def __init__(self, robot_ids, initial_robots_poses=None, safety_layer=True, safety_radius=None):
        # constants
        # robots
        self.ROBOT_IDS = robot_ids
        self.N = len(self.ROBOT_IDS)
        # time
        self.TIMEOUT_SET_MOBILE_BASE_SPEED = 50 # milliseconds (RobotHardware frequency set to 20Hz)
        self.TIMEOUT_GET_POSES = 10 # milliseconds (Vicon frequency set to 100Hz)
        self.DT = (self.TIMEOUT_SET_MOBILE_BASE_SPEED + self.TIMEOUT_GET_POSES) / 1000.
        # robot control
        self.MAX_LINEAR_SPEED = 0.1 # meters / second
        self.MAX_ANGULAR_SPEED = 30 * np.pi / 180 # radians / second
        self.SAFETY_LAYER = safety_layer
        self.GAMMA = 10.
        self.P = np.eye(2 * self.N)
        # dimensions
        self.ENV = [-2., -2., 4., 4.] # (x, y) can vary from (ENV[0], ENV[1]) to (ENV[0]+ENV[2], ENV[1]+ENV[3])
        self.ROBOT_SIZE = [0.24, 0.32] # [w, l]
        if safety_radius is None:
            self.SAFETY_RADIUS = max(self.ROBOT_SIZE) * 2.2
        else:
            self.SAFETY_RADIUS = safety_radius
        self.GRIPPER_SIZE = 0.1
        self.INIT_POSES_THRESH = 0.5

        # initialize positions of the robots
        self.intialize_robots_to_poses(initial_robots_poses)

    @abstractmethod
    def intialize_robots_to_poses(self, initial_robots_poses):
        pass

    @abstractmethod
    def set_robots_speeds_and_grippers_powers(self, robots_speeds, robots_grippers_powers, local_frame=False):
        pass

    @abstractmethod
    def set_robots_arms_poses(self, robots_arms_poses, wait_s=2.6):
        pass

    @abstractmethod
    def set_leds(self, robots_leds):
        pass

    @abstractmethod
    def get_robots_poses(self, check_all_poses_received=False):
        pass

    def safe_and_max_speeds(self, robots_speeds):
        for i in range(self.N):
            ui_norm = np.linalg.norm(robots_speeds[:2, i])
            if ui_norm > self.MAX_LINEAR_SPEED:
                robots_speeds[:2, i] = robots_speeds[:2, i] / ui_norm * self.MAX_LINEAR_SPEED
            robots_speeds[2, :] = np.clip(robots_speeds[2, :], -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)
        if self.SAFETY_LAYER:
            q = -2. * robots_speeds[:2, :].T.reshape((1, 2 * self.N))
            Aqp = np.zeros((int(self.N * (self.N - 1) / 2), 2 * self.N))
            bqp = np.zeros((int(self.N * (self.N - 1) / 2), ))
            constraint_idx = 0
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    Aqp[constraint_idx, 2 * i : 2 * (i + 1)] = -2. * (self.robots_poses[:2, i] - self.robots_poses[:2, j])
                    Aqp[constraint_idx, 2 * j : 2 * (j + 1)] = 2. * (self.robots_poses[:2, i] - self.robots_poses[:2, j])
                    bqp[constraint_idx] = self.GAMMA * (np.linalg.norm(self.robots_poses[:2, i] - self.robots_poses[:2, j]) ** 2 - self.SAFETY_RADIUS ** 2)
                    constraint_idx = constraint_idx + 1
            u = solve_qp(2 * self.P, q.T, Aqp, bqp, solver='osqp')
            robots_speeds[:2, :] = u.reshape((self.N, 2)).T
        return robots_speeds
    
    @staticmethod
    def transform_velocity_global_to_local(robots_speeds, theta):
        # robots_speeds : np.array, 3 x N
        # theta : np.array, 1 x N
        N = robots_speeds.shape[1]
        robots_speeds_local = np.zeros((3, N))
        for i in range(N):
            x_dot = robots_speeds[0, i]
            y_dot = robots_speeds[1, i]
            c_th = cos(theta[i])
            s_th = sin(theta[i])
            robots_speeds_local[:2, i] = np.array([c_th * x_dot + s_th * y_dot, -s_th * x_dot + c_th * y_dot])
        robots_speeds_local[2, :] = robots_speeds[2, :]
        return robots_speeds_local

    @staticmethod
    def transform_velocity_local_to_global(robots_speeds, theta):
        # robots_speeds : np.array, 3 x N
        # theta : np.array, 1 x N
        N = robots_speeds.shape[1]
        robots_speeds_global = np.zeros((3, N))
        for i in range(N):
            x_dot = robots_speeds[0, i]
            y_dot = robots_speeds[1, i]
            c_th = cos(theta[i])
            s_th = sin(theta[i])
            robots_speeds_global[:2, i] = np.array([c_th * x_dot - s_th * y_dot, s_th * x_dot + c_th * y_dot])
        robots_speeds_global[2, :] = robots_speeds[2, :]
        return robots_speeds_global