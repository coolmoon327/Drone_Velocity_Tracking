import numpy as np
import pybullet as p
from gymnasium import spaces
from ultralytics import YOLO
import torch

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

PENALTY = -1000.
MAX_DIS = 5.
MAX_V = 1.

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
                 output_folder='results',
                 debug=False
                 ):
        if initial_xyzs is None:
            initial_xyzs = np.array([[0., 0., .8]])
        # Keep the TARGET_DIS from the TARGET_POS
        self.TARGET_rpy = np.zeros(3)
        self.TARGET_pos = np.zeros(3)
        self.TARGET_vel = np.zeros(4)
        self.TARGET_dis = .4
        self.EPISODE_LEN_SEC = 8
        self.IMGs_per_step = 2
        self.IMG_RES = np.array([640, 480]) 
        self.IMG_FRAME_PER_SEC = ctrl_freq
        self.debug = debug

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
        self.IMGs_steps = []
        self.IMGs_features = []
        self.step_num = 0

        self.reward_penalty = 0.
        self.last_kin = None
        self.obs = None
        self.last_obs = None
        self.lost_targets = 0
        self.model = YOLO('yolov8n.pt')

    ### Env API #############################################################################

    def reset(self,
            seed : int = None,
            options : dict = None):
        
        if self.debug:
            print("============ RESET ============")
    
        self.reward_penalty = 0.
        self.last_kin = self._getDroneStateVector(0)
        self.last_obs = None
        self.lost_targets = 0

        self.IMGs.clear()
        self.IMGs_steps.clear()
        self.IMGs_features.clear()
        self.lost_targets = 0
        self.step_num = 0

        self.reward_penalty = 0.

        super().reset(seed=seed, options=options)

        if seed is not None:
            np.random.seed(seed)

        self._setTarget()

        state = self._getDroneStateVector(0)
        self.TARGET_rpy = state[7:10]

        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info

    def step(self,
             action
             ):
        self.IMGs.clear()
        self.IMGs_steps.clear()
        self.IMGs_features.clear()
        self.step_num += 1

        self.reward_penalty = 0.
        self.last_kin = self._getDroneStateVector(0)
        self.last_obs = None

        ret = super().step(action=action)
        self._updateTarget()

        return ret
    
    ### Env Obj #############################################################################

    def _setTarget(self):
        self.TARGET_pos = np.array([np.random.randint(20,200)/100., 0., .5])
        self.TARGET_vel = np.array([np.random.randint(-10,10)/100., 0., 0.])

        obj_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                        fileName="gym_pybullet_drones/coolmoon/models/SittingBaby/baby.obj",
                                        meshScale=[0.02, 0.02, 0.02])
        
        self.TARGET_ID = p.createMultiBody(baseMass=1.0,
                                        baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=obj_visual_shape_id,
                                        basePosition=self.TARGET_pos,
                                        baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2]))
        
        # No gravity or inertial
        p.changeDynamics(self.TARGET_ID, -1, mass=0.)

        p.resetBaseVelocity(self.TARGET_ID, linearVelocity=self.TARGET_vel)
        
    def _updateTarget(self):
        pass

    def _addObstacles(self):
        pass

    ### State #############################################################################

    def _observationSpace(self):
        return spaces.Box(low=0,
                        high=1.,
                        shape=(1, 10), dtype=np.uint8)   # last xywh, last action, now xywh, now velocity
    
    def _computeObs(self):
        self.last_obs = self.obs
        
        # TODO: 分情况讨论，某一个为空的情况
        # 前一个 lost 了，则 lost_targets > 0，controller 回退到根据当前目标大小调整距离
        # 看一下怎么针对性修改 reward
        pass

        rgb, _, _ = self._getDroneImages(0, segmentation=False)
        rgb = rgb[:,:,:3]

        if torch.cuda.is_available():
            results = self.model.predict(rgb, device=torch.device("cuda:2"), max_det=1, classes=[0], verbose=False)
        else:
            results = self.model.predict(rgb, max_det=1, classes=[0], verbose=False)

        lose_target = False
        for r in results:
            xywhn = r.boxes.xywhn.cpu()
            if xywhn.numel() == 0:
                lose_target = True
                break
            if self.debug:
                print("Target XYWH:", xywhn, ", area:", xywhn[0][2]*xywhn[0][3]*self.IMG_RES[0]*self.IMG_RES[1])
        if lose_target:
            if self.debug:
                print("==== Warning: Lose target! ====")
            self.lost_targets += 1
            obs = np.zeros([1, 4])
        else:
            self.lost_targets = 0
            obs = xywhn

        now_kin = self._getDroneStateVector(0)
        cur_vel = np.array(now_kin[10:13])
        
        if abs(cur_vel[0]) != 0:
            velocity = np.linalg.norm(cur_vel[0:3]) * (cur_vel[0]/abs(cur_vel[0]))
        else:
            velocity = 0.

        # TODO: 检查 obs 是不是对的
        if self.last_obs is not None:
            last_obs = self.obs
        else:
            last_obs = np.zeros((1, 10))
        self.obs = np.concatenate((last_obs[0, 5:9], self.action_buffer[-1][0][0], obs, velocity), axis=None).reshape(1, -1)
        self.obs = np.array(self.obs).astype('float32')
        # print("OBS:", self.obs)

        return self.obs
    
    ### Action #############################################################################

    def _actionSpace(self):
        if self.ACT_TYPE == ActionType.VEL:
            # only velocity increase (0, 1), will be magnified by MAX_V
            size = 1
            act_lower_bound = np.array([0*np.ones(size) for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        else:
            return super()._actionSpace()

    def _preprocessAction(self, action):
        if self.lost_targets == 0:
            if np.any(self.obs[0, 0:3] != 0):
                vel = action[0][0] * MAX_V  # (0., 1.) be magnified by MAX_V

                if self.obs[0][2] > 0.27:
                    neg = -1
                else:
                    neg = 1
                
                action = np.array([[vel * neg]])
                rpm = self._x_increase_to_rpm(action)

                if self.debug:
                    print("Action", action, ", RPM:", rpm)
                    
                return rpm
            
        rpm = self._vel_to_rpm(np.zeros([1,4]))
        # rpm = self._x_increase_to_rpm(np.zeros([1,1]))        # 会持续当前速度

        return rpm

    def _vel_to_rpm(self, vel):
        ## vel: np.array(1,4) ##

        self.action_buffer.append(vel)
        vel = vel[0, :]
        rpm = np.zeros((self.NUM_DRONES,4))
        state = self._getDroneStateVector(0)

        if np.linalg.norm(vel[0:3]) != 0:
            v_unit_vector = vel[0:3] / np.linalg.norm(vel[0:3])
        else:
            v_unit_vector = np.zeros(3)
        temp, _, _ = self.ctrl[0].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                cur_pos=state[0:3],
                                                cur_quat=state[3:7],
                                                cur_vel=state[10:13],
                                                cur_ang_vel=state[13:16],
                                                target_pos=np.array([state[0], self.INIT_XYZS[0][1], self.INIT_XYZS[0][2]]), # same as the current X position
                                                target_rpy=self.TARGET_rpy, # keep original rpy
                                                target_vel=self.SPEED_LIMIT * np.abs(vel[3]) * v_unit_vector # target the desired velocity vector
                                                )
        rpm[0,:] = temp
        return rpm

    def _x_increase_to_rpm(self, x_increase):
        ## x_increase: np.array(1,1) ##

        self.action_buffer.append(x_increase)
        x_increase = x_increase[0, 0]
        rpm = np.zeros((self.NUM_DRONES,4))
        state = self._getDroneStateVector(0)

        temp, _, _ = self.ctrl[0].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                cur_pos=state[0:3],
                                                cur_quat=state[3:7],
                                                cur_vel=state[10:13],
                                                cur_ang_vel=state[13:16],
                                                target_pos=np.array([state[0] + x_increase, self.INIT_XYZS[0][1], self.INIT_XYZS[0][2]]),    # 不修改速度的方向避免误差累积
                                                target_rpy=self.TARGET_rpy, # 不使用当前 RPY，避免误差累积
                                                target_vel=state[10:13] # 不使用 velocity 修改速度
                                                )
        rpm[0,:] = temp
        return rpm

    ### Reward #############################################################################
    
    # state = self._getDroneStateVector(drone_id)
    # cur_pos=state[0:3]
    # cur_quat=state[3:7]
    # cur_vel=state[10:13]
    # cur_ang_vel=state[13:16]

    def _computeDis(self, kin_state):
        cur_pos=kin_state[0:3]
        distance = np.linalg.norm(self.TARGET_pos-cur_pos)
        return distance

    def _compareVel(self, kin_state):
        cur_vel = np.array(kin_state[10:13])
        return np.linalg.norm(cur_vel - self.TARGET_vel)

    def _computeReward(self):
        if self.lost_targets:
            # 丢失目标后无人机不会执行 PPO action, 不应该进行任何惩罚/奖励
            return 0.

        w = self.obs[0][2]
        ret = 1. / (np.abs(0.27 - w)+ 0.01) - 10.

        # TODO: 看看是否加入速度相关的奖励

        if self.debug:
            print("Reward:", ret, ", W:", w)


        if self._computeTerminated():
            ret += 100.
        else:
            if abs(self.action_buffer[-1][0][0]) < 1e-4 and self.lost_targets == 0:
                # 惩罚摆烂不动
                self.reward_penalty += PENALTY / 4.
                if self.debug:
                    print("No Moving.")

        ret += self.reward_penalty

        return ret

    ### Info #############################################################################
    
    def _computeTerminated(self):
        now_kin = self._getDroneStateVector(0)
        distance = self._computeDis(now_kin)
        # dis_vel = self._compareVel(now_kin)

        # TODO: 先试试不限制速度，直接跟随

        # if np.abs(self.TARGET_dis - distance) < .01 and dis_vel < .01:
        if np.abs(self.TARGET_dis - distance) < .01:
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

        if np.abs(self.TARGET_dis - self._computeDis(self._getDroneStateVector(0))) > MAX_DIS:
            # Truncate when the drone is too far away
            if self.debug:
                print("Truncated: too far.")
            self.reward_penalty += PENALTY
            return True
        
        if np.abs(state[7]) > .9 or np.abs(state[8]) > .9:
            # Truncate when the drone is too tilted
            if self.debug:
                print(f"Truncated: too titlted: {state[7:9]}")
            self.reward_penalty += PENALTY
            return True
        
        if self.lost_targets > 1:
            # Target lost
            if self.debug:
                print("Truncated: lose target.")
            self.reward_penalty += PENALTY
            return True

        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
    
    def _computeInfo(self):
        return {}
