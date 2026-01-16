import sys
sys.path.append('../')
from multi_dji_robomaster_ep_sim import MultiDJIRoboMasterEPSim
import numpy as np

ROBOT_IDS = list(range(1,11))
N = len(ROBOT_IDS)
INITIAL_ROBOTS_POSES = np.vstack((3.5 * np.ones((1, N)), 2. * np.ones((1, N)), -np.pi / 2. * np.ones((1, N))))

mrs = MultiDJIRoboMasterEPSim(ROBOT_IDS,
                                initial_robots_poses=INITIAL_ROBOTS_POSES,
                                safety_layer=True,
                                safety_radius=0.5)