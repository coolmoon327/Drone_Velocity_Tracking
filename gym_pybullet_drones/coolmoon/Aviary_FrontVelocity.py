import numpy as np
import pybullet as p
from gymnasium import spaces
from ultralytics import YOLO
import torch

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

class Aviary_FrontVelocity(BaseRLAviary):
    """Env for front velocity control"""

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
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)         # no control panel

        if initial_xyzs is None:
            initial_xyzs = np.array([[0., 0., .8]])
        # Keep the TARGET_DIS from the TARGET_POS
        self.TARGET_pos = np.zeros(3)
        self.TARGET_vel = np.zeros(4)
        self.TARGET_dis = 1.
        self.EPISODE_LEN_SEC = 8
        self.IMGs_per_step = 2
        self.IMG_RES = np.array([1280, 640])    # check if it works

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=ObservationType('kin'),    # no matter what kind of obs
                         act=act,
                         output_folder=output_folder
                         )

        self.IMGs = []
        self.model = YOLO('yolov8n.pt')
        self.has_target = False
        self.step_num = 0

    ### Env API #############################################################################

    def reset(self,
            seed : int = None,
            options : dict = None):
        ret = super().reset(seed=seed, options=options)
        np.random.seed(seed)
        self._setTarget()

        self.last_rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))

        self.has_target = False
        self.step_num = 0

        return ret

    def step(self,
             action
             ):
        ret = super().step(action=action)
        self._updateTarget()

        self.has_target = False
        self.step_num += 1

        return ret

    def _stepToNextControl(self, clipped_action):
        #### Repeat for as many as the aggregate physics steps #####
        self.IMGs.clear()

        for i in range(self.PYB_STEPS_PER_CTRL):
            self._stepSimulation(clipped_action)

            if (self.PYB_STEPS_PER_CTRL - i - 1) % (self.PYB_STEPS_PER_CTRL / self.IMGs_per_step) == 0:
                rgb, _, _ = self._getDroneImages(0, segmentation=False)
                self.IMGs.append(rgb[:,:,:3])   # from (h,w,4) to (h,w,3)

    ### Env Obj #############################################################################

    def _setTarget(self):
        if self.GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)   # no rendering when loading

        self.TARGET_pos = np.array([np.random.randint(100,300)/100., 0., .5])
        self.TARGET_vel = np.array([np.random.randint(-100,100)/50., 0., 0.])

        obj_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                           fileName="gym_pybullet_drones/coolmoon/models/14-girl-obj/girl OBJ.obj",
                                           meshScale=[0.3, 0.3, 0.3],
                                           rgbaColor=[1.0, 0.8, 0.6, 1.0])

        self.TARGET_ID = p.createMultiBody(baseMass=1.0,
                                        baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=obj_visual_shape_id,
                                        basePosition=self.TARGET_pos,
                                        baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, -np.pi/2]))
        
        p.resetBaseVelocity(self.TARGET_ID, linearVelocity=self.TARGET_vel)

        # No gravity or inertial
        p.changeDynamics(self.TARGET_ID, -1, mass=0.)
        
        if self.GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        
    def _updateTarget(self):
        pass

    def _addObstacles(self):
        pass

    ### State #############################################################################

    def _observationSpace(self):
        # targets' xywhn of self.IMGs_per_step images
        return spaces.Box(low=0,
                            high=1.,
                            shape=(self.IMGs_per_step, 4), dtype=np.uint8)
    
    def _computeObs(self):
        obs = []

        if len(self.IMGs) == 0:
            # TODO: 检查初始状态
            rgb, _, _ = self._getDroneImages(0, segmentation=False)
            self.IMGs.append(rgb[:,:,:3])
            obs.append(np.zeros(4))

        # 用 _stepToNextControl 收集的照片 self.IMGs 整理出 obs
        results = self.model.predict(self.IMGs, device=torch.device("cuda:3"), max_det=1, classes=[0])

        for r in results:
            xywhn = r.boxes.xywhn.cpu()
            if xywhn.numel() == 0:
                obs.append(np.zeros(4))
            else:
                obs.append(xywhn.numpy()[0])
                self.has_target = True

        obs = np.array([obs]).astype('float32')

        # print(obs)

        return obs
    
    ### Action #############################################################################
    
    def _actionSpace(self):
        if self.ACT_TYPE == ActionType.VEL:
            # only velocity
            size = 1
            act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        else:
            return super()._actionSpace()

    def _preprocessAction(self, action):
        # print(action)
        if self.ACT_TYPE == ActionType.VEL:
            # restore the format of env's action
            if action.shape[0] != 1:
                print("Warning: there are more than 1 action.")
            drone_id = 0
            target = action[drone_id, :]
            # 1. get current state
            state = self._getDroneStateVector(drone_id)
            cur_vel=state[10:13]

            # 2. keep current direction
            # action = np.array([cur_vel+action[0]])
            # TODO: 怎么算出速度大小
            speed = np.linalg.norm(cur_vel) + action[0][0]
            action = np.hstack((cur_vel, speed))
            action = np.expand_dims(action, axis=0)
            # print(cur_vel, speed, action)
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

    def _computeReward(self):
        distance = self._computeDis()
        # dis_vel = self._compareVel()

        # ret = 10 / (np.abs(self.TARGET_dis - distance) + 1e-4) - dis_vel
        ret = -np.abs(self.TARGET_dis - distance)

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
        
        if self.step_num > 10 and self.has_target == False:
            # 目标脱离摄像头
            return True

        # TODO: 怎么判断无人机掉在了地上

        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
    
    def _computeInfo(self):
        return {}
