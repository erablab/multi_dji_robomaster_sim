import sys
sys.path.append('../')
from multi_dji_robomaster_ep_sim import MultiDJIRoboMasterEPSim
import numpy as np
from math import cos, sin

ROBOT_IDS = [8, 7, 3, 1, 2, 9, 4, 5, 6]
N = len(ROBOT_IDS)
INITIAL_ROBOTS_POSES = np.vstack((np.linspace(-2., 2., N), np.zeros((1, N)), np.pi * np.ones((1, N))))

mrs = MultiDJIRoboMasterEPSim(ROBOT_IDS,
                            backend_server_ip=None,
                            initial_robots_poses=INITIAL_ROBOTS_POSES,
                            safety_layer=True,
                            safety_radius=0.5)

for t in np.arange(0., 10., mrs.DT):
    q = mrs.get_robots_poses()
    print('robot poses', q)

    robots_speeds = np.vstack((1.*np.ones((1, N)), 1.*np.ones((1, N)), 1.*np.ones((1, N)))) # v_x, v_y, omega_z in a global reference frame OR longitudinal, lateral, angular in a local reference frame
    robots_grippers_powers = np.random.random((1, N))
    robots_leds = np.random.randint(0, 255, (3, N))
    robots_arms_poses = np.random.random((2, N))

    mrs.set_robots_speeds_and_grippers_powers(robots_speeds, robots_grippers_powers, local_frame=False)
    mrs.set_leds(robots_leds)
    mrs.set_robots_arms_poses(robots_arms_poses, wait_s=5.0)