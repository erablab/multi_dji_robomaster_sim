from multi_dji_robomaster_ep_abc import MultiDJIRoboMasterEP
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math
import time

class MultiDJIRoboMasterEPSim(MultiDJIRoboMasterEP):
    def __init__(self, robot_ids, backend_server_ip=None, initial_robots_poses=None, safety_layer=True, safety_radius=None):
        super().__init__(robot_ids, initial_robots_poses, safety_layer, safety_radius)
        
    def intialize_robots_to_poses(self, initial_robots_poses):
        self.robots_poses = np.vstack((self.ENV[0] + self.ENV[2] * np.random.random((1, self.N)), self.ENV[1] + self.ENV[3] * np.random.random((1, self.N)), 2 * np.pi * np.random.random((1, self.N))))

        # init plot
        self.figure = []
        self.axes = []
        self.patches_robots = []
        self.patches_grippers = []
        self.text_ids = []
        self.__init_plot()

        if initial_robots_poses is not None:
            print('Initializing robots to specified initial poses ...')
            pose_error = initial_robots_poses - self.robots_poses
            pose_error[2, :] = np.arctan2(np.sin(pose_error[2, :]), np.cos(pose_error[2, :]))
            while np.linalg.norm(pose_error) > self.INIT_POSES_THRESH:
                pose_error = initial_robots_poses - self.robots_poses
                pose_error[2, :] = np.arctan2(np.sin(pose_error[2, :]), np.cos(pose_error[2, :]))
                robots_speeds = 10. * pose_error
                self.set_robots_speeds_and_grippers_powers(robots_speeds, np.zeros((1, self.N)))
            self.set_leds(np.tile(np.array([[173], [216], [230]], dtype=int), (1, self.N)))
            self.__update_plot()
            print('... done (LEDs should be blue now)')
            time.sleep(1.)
    
    def __init_plot(self):
        self.figure, self.axes = plt.subplots()
        p_env = patches.Rectangle(np.array([self.ENV[0], self.ENV[1]]), self.ENV[2], self.ENV[3], edgecolor=(0, 0, 0, 1), fill=False, linewidth=4)
        self.axes.add_patch(p_env)

        for i in range(self.N):
            R = np.array([[math.cos(self.robots_poses[2, i]), -math.sin(self.robots_poses[2, i])], [math.sin(self.robots_poses[2, i]), math.cos(self.robots_poses[2, i])]])
            t = np.array([self.robots_poses[0, i], self.robots_poses[1, i]])
            p_robot = patches.Polygon(t + (np.array([[self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                                     [-self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                                     [-self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
                                                     [self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0]]) @ R.T),
                                                     facecolor='k')
            p_gripper = patches.Polygon(t + (np.array([[self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, 0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, 0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, -0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0]]) @ R.T),
                                                       facecolor='k')
            text_id = plt.text(self.robots_poses[0, i], self.robots_poses[1, i], str(self.ROBOT_IDS[i]))
            self.patches_robots.append(p_robot)
            self.patches_grippers.append(p_gripper)
            self.text_ids.append(text_id)
            self.axes.add_patch(p_robot)
            self.axes.add_patch(p_gripper)
        
        self.axes.set_xlim(self.ENV[0] - max(self.ROBOT_SIZE), self.ENV[0] + self.ENV[2] + max(self.ROBOT_SIZE))
        self.axes.set_xlim(self.ENV[1] - max(self.ROBOT_SIZE), self.ENV[1] + self.ENV[3] + max(self.ROBOT_SIZE))
        self.axes.grid()
        # self.axes.set_axis_off()
        self.axes.axis('equal')

        plt.ion()
        plt.show()

    def __update_plot(self):
        for i in range(self.N):
            R = np.array([[math.cos(self.robots_poses[2, i]), -math.sin(self.robots_poses[2, i])], [math.sin(self.robots_poses[2, i]), math.cos(self.robots_poses[2, i])]])
            t = np.array([self.robots_poses[0, i], self.robots_poses[1, i]])
            xy_robot = t + (np.array([[self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                      [-self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                      [-self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
                                      [self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0]]) @ R.T)
            xy_gripper = t + (np.array([[self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, 0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, 0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, -0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0]]) @ R.T)
        
            self.patches_robots[i].xy = xy_robot
            self.patches_grippers[i].xy = xy_gripper
            self.text_ids[i].set_position((self.robots_poses[0, i], self.robots_poses[1, i]))

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def set_robots_speeds_and_grippers_powers(self, robots_speeds, robots_grippers_powers, local_frame=False):
        if local_frame is True:
            robots_speeds = MultiDJIRoboMasterEP.transform_velocity_local_to_global(robots_speeds, self.robots_poses[2, :])
        robots_speeds = self.safe_and_max_speeds(robots_speeds)
        # for the simulator we don't need to transform back to local since we're integrating directly in the global frame
        for i in range(self.N):
            self.robots_poses[0, i] = self.robots_poses[0, i] + robots_speeds[0, i] * self.DT
            self.robots_poses[1, i] = self.robots_poses[1, i] + robots_speeds[1, i] * self.DT
            self.robots_poses[2, i] = self.robots_poses[2, i] + robots_speeds[2, i] * self.DT
        self.robots_poses[2, :] = np.arctan2(np.sin(self.robots_poses[2, :]), np.cos(self.robots_poses[2, :]))

        # grippers motion not implemented

        # update plot
        self.__update_plot()

    def set_robots_arms_poses(self, robots_arms_poses, wait_s=2.6):
        # arms motion not implemented
        pass

    def set_leds(self, robots_leds):
        for i in range(self.N):
            self.patches_robots[i].set_facecolor(robots_leds[:, i] / 255.)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
    
    def get_robots_poses(self, check_all_poses_received=False):
        return self.robots_poses - np.vstack((self.ROBOT_SIZE[1] / 2.0 * np.cos(self.robots_poses[2, :]),
                                self.ROBOT_SIZE[1] / 2.0 * np.sin(self.robots_poses[2, :]),
                                np.zeros((1, self.N))))