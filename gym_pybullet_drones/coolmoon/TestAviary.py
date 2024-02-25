import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

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
        if initial_xyzs is None:
            initial_xyzs = np.array([[0., 0., .8]])
        # Keep the TARGET_DIS from the TARGET_POS
        self.TARGET_pos = np.zeros(3)
        self.TARGET_vel = np.zeros(4)
        self.TARGET_dis = 1.
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

    ### Env API #############################################################################

    def reset(self,
            seed : int = None,
            options : dict = None):
        ret = super().reset(seed=seed, options=options)
        np.random.seed(seed)
        self._setTarget()

        self.last_rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))

        return ret

    def step(self,
             action
             ):
        ret = super().step(action=action)
        self._updateTarget()
        return ret

    ### Env Obj #############################################################################

    def _setTarget(self):
        # self.TARGET_POS = np.array([np.random.randint(100,500)/100., np.random.randint(-300,300)/100., 1])
        self.TARGET_pos = np.array([np.random.randint(100,300)/100., 0., .5])
        self.TARGET_vel = np.array([np.random.randint(-100,100)/100., 0., 0.])

        self.TARGET_ID = p.loadURDF("human-gazebo/humanSubject01/humanSubject01_66dof.urdf",
                    self.TARGET_pos,
                    p.getQuaternionFromEuler([0, 0, np.pi]),
                    physicsClientId=self.CLIENT,
                    globalScaling=0.3
                    )
        p.resetBaseVelocity(self.TARGET_ID, linearVelocity=self.TARGET_vel)

        # No gravity or inertial
        p.changeDynamics(self.TARGET_ID, -1, mass=0.)
        for jointIndex in range(p.getNumJoints(self.TARGET_ID)):
            p.changeDynamics(self.TARGET_ID, jointIndex, mass=0.)

        
    def _updateTarget(self):
        pass
        # pos, orn = p.getBasePositionAndOrientation(self.TARGET_ID)
        # time_slot_length = 1. / self.CTRL_FREQ
        # new_pos = np.array(pos) + self.TARGET_vel * time_slot_length
        # p.resetBasePositionAndOrientation(self.TARGET_ID, new_pos, orn)

    def _addObstacles(self):
        pass
        # item_pos = [np.random.randint(0,200)/100., np.random.randint(-100,100)/100., 0]

        # p.loadURDF("duck_vhacd.urdf",
        #             item_pos,
        #             p.getQuaternionFromEuler([0, 0, 0]),
        #             physicsClientId=self.CLIENT
        #             )

    ### State #############################################################################

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.RGB:
            # (self.IMG_RES[1], self.IMG_RES[0], 4) x 2, now & last
            return spaces.Box(low=0,
                            high=255,
                            shape=(self.IMG_RES[1], self.IMG_RES[0], 8), dtype=np.uint8)
        return super()._observationSpace()
    
    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                self.last_rgb = self.rgb
                self.rgb[0], self.dep[0], self.seg[0] = self._getDroneImages(0, segmentation=False)
                #### Printing observation to PNG frames example ############
                if self.RECORD:
                    self._exportImage(img_type=ImageType.RGB,
                                        img_input=self.rgb[0],
                                        path=self.ONBOARD_IMG_PATH+"/drone_0",
                                        frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                        )
            concatenated_img = np.concatenate((self.rgb[0], self.last_rgb[0]), axis=2)
            return np.array(concatenated_img).astype('float32')
        return super()._computeObs()
    
    ### Action #############################################################################
    
    def _actionSpace(self):
        if self.ACT_TYPE == ActionType.VEL:
            # only x & y + velocity
            size = 3
            act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        else:
            return super()._actionSpace()

    def _preprocessAction(self, action):
        if self.ACT_TYPE == ActionType.VEL:
            # restore the format of env's action
            if action.shape[0] != 1:
                print("Warning: there are more than 1 action.")
            drone_id = 0
            target = action[drone_id, :]
            # 1. get current state
            state = self._getDroneStateVector(drone_id)
            cur_vel=state[10:13]
            # 2. append z into the action
            action = np.insert(action, 2, cur_vel[2], axis=1)
        return super()._preprocessAction(action)


    ### Reward #############################################################################
    
    # state = self._getDroneStateVector(drone_id)
    # cur_pos=state[0:3]
    # cur_quat=state[3:7]
    # cur_vel=state[10:13]
    # cur_ang_vel=state[13:16]

    def _computeDis(self):
        state = self._getDroneStateVector(0)
        cur_pos=state[0:3]
        distance = np.linalg.norm(self.TARGET_pos-cur_pos)
        return distance

    def _compareVel(self):
        state = self._getDroneStateVector(0)
        cur_vel = np.array(state[10:13])
        return np.linalg.norm(cur_vel - self.TARGET_vel)

        # cur_speed = np.linalg.norm(cur_vel)
        # cur_vector = cur_vel / np.linalg.norm(cur_speed)

        # target_speed = np.linalg.norm(self.TARGET_vel)
        # target_vector = self.TARGET_vel / target_speed

        # # cosine_similarity = np.dot(cur_vector, target_vector)
        # # angle_diff = np.arccos(cosine_similarity)

        # euclidean_distance = np.linalg.norm(cur_vector - target_vector)

        # return np.abs(euclidean_distance) + np.abs(cur_speed - target_speed)

    def _computeReward(self):
        distance = self._computeDis()
        dis_vel = self._compareVel()

        ret = 10 / (np.abs(self.TARGET_dis - distance) + 1e-4) - dis_vel

        if self._computeTerminated():
            ret += 100.

        if self._computeTruncated():
            ret += -100.

        return ret

    ### Info #############################################################################
    
    def _computeTerminated(self):
        distance = self._computeDis()
        dis_vel = self._compareVel()
        if np.abs(self.TARGET_dis - distance) < .01 and dis_vel < .01:
            return True
        else:
            return False
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)

        if np.abs(self.TARGET_dis - self._computeDis()) > 10.:
            # Truncate when the drone is too far away
            return True
        
        if np.abs(state[7]) > .4 or np.abs(state[8]) > .4:
            # Truncate when the drone is too tilted
            return True
        
        # TODO: 怎么判断无人机掉在了地上

        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
    
    def _computeInfo(self):
        return {}
