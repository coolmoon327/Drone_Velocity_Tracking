import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TestAviary(BaseRLAviary):
    """测试用环境"""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.VEL,
                 output_folder='results'
                 ):
        # Keep the TARGET_DIS from the TARGET_POS
        self.TARGET_POS = np.zeros(3)
        self.TARGET_DIS = 1.
        self.EPISODE_LEN_SEC = 8
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         output_folder=output_folder
                         )

    ################################################################################

    def _addObstacles(self):
        item_pos = [np.random.randint(0,200)/100., np.random.randint(-100,100)/100., 0]
        self.TARGET_POS = np.array(item_pos)

        p.loadURDF("duck_vhacd.urdf",
                    item_pos,
                    p.getQuaternionFromEuler([0, 0, 0]),
                    physicsClientId=self.CLIENT
                    )

    ################################################################################
    
    def _computeDis(self):
        state = self._getDroneStateVector(0)
        distance = np.linalg.norm(self.TARGET_POS-state[0:3])
        return distance

    def _computeReward(self):
        distance = self._computeDis()
        ret = 1. - (self.TARGET_DIS - distance)**2
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        distance = self._computeDis()
        if (self.TARGET_DIS - distance)**2 < .0001:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        return {}
