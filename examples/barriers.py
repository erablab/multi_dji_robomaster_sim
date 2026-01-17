import sys
sys.path.append('../')
from multi_dji_robomaster_ep_sim import MultiDJIRoboMasterEPSim
import numpy as np
import scipy.spatial.distance as distances
import time

# constants
LOOP_TIME = 0.05
ROBOT_IDS = list(range(1, 11))
N = len(ROBOT_IDS)

RADIUS = 2.0
THETAS = np.linspace(0, 2*np.pi * (N - 1) / float(N), N)
TARGET_ROBOTS_POSES_0 = np.vstack((RADIUS * np.cos(THETAS), 
                                RADIUS * np.sin(THETAS),
                                np.arctan2(np.sin(THETAS + np.pi), np.cos(THETAS + np.pi))))
TARGET_ROBOTS_POSES_1 = np.vstack((RADIUS * np.cos(THETAS + np.pi),
                                RADIUS * np.sin(THETAS + np.pi),
                                np.arctan2(np.sin(THETAS + np.pi), np.cos(THETAS + np.pi))))

SAFETY_DIST = 0.8

RED = np.tile(np.array([[255], [0], [0]], dtype=int), (1, N))
GREEN = np.tile(np.array([[0], [255], [0]], dtype=int), (1, N))
def interpolate_red_green(poses):
    d = np.min(100. * np.eye(poses.shape[1]) + distances.cdist(poses[:2, :].T, poses[:2, :].T), axis=0)
    l = (d - SAFETY_DIST) / (RADIUS * 2 * np.sin(2 * np.pi / N / 2) - SAFETY_DIST)
    return (np.clip((1 - l) * RED + l * GREEN, 0, 255)).astype(int)

# initialization
target_robots_poses = TARGET_ROBOTS_POSES_0
target_flag = 0

mrs = MultiDJIRoboMasterEPSim(ROBOT_IDS,
                            backend_server_ip=None,
                            initial_robots_poses=target_robots_poses,
                            safety_layer=True,
                            safety_radius=SAFETY_DIST)

# main control loop
while True:
    loop_start = time.time()

    q = mrs.get_robots_poses()

    pose_diff = target_robots_poses - q
    pose_diff[2, :] = np.arctan2(np.sin(pose_diff[2, :]), np.cos(pose_diff[2, :]))
    robots_dist = np.linalg.norm(pose_diff)
    
    if robots_dist < mrs.INIT_POSES_THRESH:
        if target_flag == 0:
            target_flag = 1
            target_robots_poses = TARGET_ROBOTS_POSES_1
        elif target_flag == 1:
            target_flag = 0
            target_robots_poses = TARGET_ROBOTS_POSES_0
    
    robots_speeds = target_robots_poses - q
    robots_speeds[2, :] = np.arctan2(np.sin(robots_speeds[2, :]), np.cos(robots_speeds[2, :]))
    robots_grippers_powers = np.zeros((N, ))
    robots_leds = interpolate_red_green(q)

    mrs.set_robots_speeds_and_grippers_powers(robots_speeds, robots_grippers_powers, local_frame=False)
    mrs.set_leds(robots_leds)

    loop_end = time.time()
    print('loop time:', loop_end - loop_start)
    if (loop_end - loop_start) < LOOP_TIME:
        time.sleep(LOOP_TIME - (loop_end - loop_start))
