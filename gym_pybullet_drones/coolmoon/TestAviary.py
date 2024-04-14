import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

from ultralytics import YOLO
import torch

PENALTY = -100.
MAX_DIS = 5.
MAX_V = 2.

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
                 output_folder='results',
                 debug=False
                 ):
        # if gui:
        #     p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)         # no control panel

        if initial_xyzs is None:
            initial_xyzs = np.array([[0., 0., .8]])
        # Keep the TARGET_DIS from the TARGET_POS
        self.TARGET_rpy = np.zeros(3)
        self.TARGET_pos = np.zeros(3)
        self.TARGET_vel = np.zeros(4)
        self.TARGET_dis = 0.4
        self.EPISODE_LEN_SEC = 8
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
                         obs=obs,
                         act=act,
                         output_folder=output_folder
                         )

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

        ret = super().reset(seed=seed, options=options)
        np.random.seed(seed)
        self._setTarget()

        state = self._getDroneStateVector(0)
        self.TARGET_rpy = state[7:10]

        return ret

    def step(self,
             action
             ):
        self.reward_penalty = 0.
        self.last_kin = self._getDroneStateVector(0)
        self.last_obs = None

        ret = super().step(action=action)
        self._updateTarget()

        return ret

    ### Env Obj #############################################################################

    def _setTarget(self):
        # if self.GUI:
        #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)   # no rendering when loading

        self.TARGET_pos = np.array([np.random.randint(20,200)/100., 0., .5])
        # self.TARGET_pos = np.array([.4, 0., .5])
        self.TARGET_vel = np.array([0., 0., 0.])
        
        # if not hasattr(self, 'TARGET_ID'):
        # 加载 OBJ 文件
        obj_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                        fileName="gym_pybullet_drones/coolmoon/models/SittingBaby/baby.obj",
                                        meshScale=[0.02, 0.02, 0.02])
        
        # 创建物体
        self.TARGET_ID = p.createMultiBody(baseMass=1.0,
                                        baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=obj_visual_shape_id,
                                        basePosition=self.TARGET_pos,
                                        baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2]))
            
        # No gravity or inertial
        p.changeDynamics(self.TARGET_ID, -1, mass=0.)

        # self.TARGET_vel = self.TARGET_vel[np.array([1, 0, 0])]
        self.TARGET_vel = np.array([0, 0, 0])

        p.resetBaseVelocity(self.TARGET_ID, linearVelocity=self.TARGET_vel)

        # for jointIndex in range(p.getNumJoints(self.TARGET_ID)):
        #     p.changeDynamics(self.TARGET_ID, jointIndex, mass=0.)
        
        # if self.GUI:
        #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        
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
        return spaces.Box(low=0,
                        high=1.,
                        shape=(1, 5), dtype=np.uint8)   # xywh, velocity
    
    def _computeObs(self):
        self.last_obs = self.obs

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


        obs = np.concatenate((obs, velocity), axis=None).reshape(1, -1)
        # print(obs)
        self.obs = np.array(obs).astype('float32')

        return self.obs
    
    ### Action #############################################################################

    def _actionSpace(self):
        if self.ACT_TYPE == ActionType.VEL:
            # only velocity (0, 1), will be magnified by MAX_V
            # 速度其实是向着某个方向的位置增量
            size = 1
            act_lower_bound = np.array([0*np.ones(size) for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
        else:
            return super()._actionSpace()

    def _preprocessAction(self, action):
        if self.obs is not None:
            if np.any(self.obs[0, 0:3] != 0):
                vel = action[0][0] * MAX_V  # (0., 1.) be magnified by MAX_V

                # neg = vel/abs(vel) if vel != 0 else 0.
                # now_kin = self._getDroneStateVector(0)
                # if (self.TARGET_pos[0] - self.TARGET_dis) < now_kin[0]:
                if self.obs[0][2] > 0.27:
                    neg = -1
                else:
                    neg = 1

                # action = np.array([[1 * neg, 0, 0, abs(vel)]])
                # rpm = self._vel_to_rpm(action) # TODO: 这里有大问题，飞的时候会漂移. 实际上，飞的时候并不能保证速度，RL 的状态还需要当前的一些 kinematics
                
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

        # cur_speed = np.linalg.norm(cur_vel)
        # cur_vector = cur_vel / np.linalg.norm(cur_speed)

        # target_speed = np.linalg.norm(self.TARGET_vel)
        # target_vector = self.TARGET_vel / target_speed

        # # cosine_similarity = np.dot(cur_vector, target_vector)
        # # angle_diff = np.arccos(cosine_similarity)

        # euclidean_distance = np.linalg.norm(cur_vector - target_vector)

        # return np.abs(euclidean_distance) + np.abs(cur_speed - target_speed)

    def _computeReward(self):
        # now_kin = self._getDroneStateVector(0)
        # distance = self._computeDis(now_kin)
        # dis_vel = self._compareVel(now_kin)

        # last_dis = self._computeDis(self.last_kin)
        # last_vel = self._compareVel(self.last_kin)

        # rew_d = (np.abs(self.TARGET_dis - last_dis) - np.abs(self.TARGET_dis - distance)) / MAX_DIS     # r = old - new
        # rew_v = (last_vel - dis_vel) / MAX_V
        # ret = 500. * rew_d + 10 * rew_v

        w = self.obs[0][2]
        if self.last_obs is not None:
            last_w = self.last_obs[0][2]
        else:
            last_w = w
        ret = 100 * (np.abs(0.27 - last_w) - np.abs(0.27 - w))

        if self.debug:
            # print("R_d, R_v, R:", rew_d, rew_v, ret)
            # print("Dis, D_v:", distance, dis_vel, "; last_Dis, last_D_v:", last_dis, last_vel)
            print("Reward:", ret, ", Last w:", last_w, ", W:", w)

        if self._computeTerminated():
            ret += 100.

        # if self._computeTruncated():
        #     ret += -100.
        ret += self.reward_penalty

        return ret

    ### Info #############################################################################
    
    def _computeTerminated(self):
        now_kin = self._getDroneStateVector(0)
        distance = self._computeDis(now_kin)
        dis_vel = self._compareVel(now_kin)

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
            # 目标脱离摄像头
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
