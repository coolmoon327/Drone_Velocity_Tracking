## Pybullet 环境交互

> 参考 `./examples/learn.py`

### Terminalogy

- Motor: 扇叶电机
- Trust: 推力
- Torque: 扭转力
- PID: Proportional-Integral-Derivative，比例-积分-微分
- RPY: Roll-Pitch-Yaw 的缩写，也称为欧拉角（Euler Angles）
- Roll: 横滚角，绕 X 轴的角度
- Pitch: 俯仰角，绕 Y 轴的角度
- Yaw: 偏航角，绕 Z 轴的角度

### Observation

#### ObservationType.RGB

- `BaseAviary.py` 的 `_getDroneImages` 返回三种图片 `rgb`, `dep`, `seg`:
    - `rgb`: (h, w, 4)-shaped array of uint8's containing the RBG(A) image captured from the n-th drone's POV.
    - `dep`: (h, w)-shaped array of uint8's containing the depth image captured from the n-th drone's POV.
    - `seg`: (h, w)-shaped array of uint8's containing the segmentation image captured from the n-th drone's POV.

- `BaseRLAviary.py` 的 `_computeObs` 使用上述的 `rgb` 作为返回值:
    - A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

- 采样频率需要精确设计
    - `IMG_CAPTURE_FREQ = int(PYB_FREQ/IMG_FRAME_PER_SEC)`
        - 默认 `PYB_FREQ=240` `IMG_FRAME_PER_SEC=24`，该值为 `10`
        - `PYB_FREQ` 是 `pybullet` 环境的更新频率，`IMG_FRAME_PER_SEC` 是摄像头的采样频率
        - 表示摄像头采用一次需要 `pybullet` 更新的次数
    - `PYB_STEPS_PER_CTRL = int(PYB_FREQ / CTRL_FREQ)`
        - 默认 `PYB_FREQ=240` `CTRL_FREQ=240`，该值为 `1`
        - `CTRL_FREQ` 是 `gym` 环境的更新频率
        - 需要满足：`PYB_FREQ % CTRL_FREQ == 0`
        - 表示 `gym` 环境更新一次需要 `pybullet` 更新的次数
    - 如果要捕获图片，需要满足: `IMG_CAPTURE_FREQ % PYB_STEPS_PER_CTRL == 0`
        - 表示 `gym` 环境更新一次时，恰好摄像头正在采样
        - 即 `CTRL_FREQ % IMG_FRAME_PER_SEC == 0`


#### ObservationType.KIN

- `BaseRLAviary.py` 的 `_computeObs` 返回 `obs = np.ndarray[float64, shape=(12,)]` 
    - 其实就是不包含 `last_clipped_action` 的 `state`，$ \{X, Y, Z, Q1, Q2, Q3, Q4, R, P, Y, VX, VY, VZ, WX, WY, WZ\} $

- `BaseAviary.py` 的 `_getDroneStateVector`  返回 `state = np.ndarray[float64, shape=(n, 20,)]`
    - `state` 列表的第一维代表 `n` 个无人机
    - 每个无人机有 20 个状态
        - `state[nth_drone][0:3]`: 当前位置 `pos`，$ \{X, Y, Z\} $
        -  `state[nth_drone][3:7]`: 当前的四元数 `quat`，$ \{Q1, Q2, Q3, Q4\} $，四元数用于描述旋转角度
        -  `state[nth_drone][7:10]`: 当前的欧拉角 `rpy`，$ \{R, P, Y\} $，代码中使用 `getEulerFromQuaternion(quat)` 得到 `rpy`
        -  `state[nth_drone][10:13]`: 当前速度 `vel`，$ \{VX, VY, VZ\} $
        -  `state[nth_drone][13:16]`: 当前角速度 `ang_v`，$ \{WX, WY, WZ\} $
        -  `state[nth_drone][16:20]`: 在 `step()` 中记录的上一次舵机 `RPM` 调整值 `last_clipped_action`



### Action

- ActionType 有四种: `RPM`, `PID`, `VEL`, `ONE_D_RPM`, `ONE_D_PID`
    - `RPM`: 通过一个四维列表分别控制四个电机
        - `target: np.ndarray[float64, shape=(4,)]`，四个电机的目标转速
    
    - `PID`: 通过 `PID` 来控制无人机前往目标位置
        - `target: np.ndarray[float64, shape=(3,)]`，目标三维空间位置

    - `VEL`: 通过 `PID` 控制无人机达到目标速度
        - `target: np.ndarray[float64, shape=(4,)]`，目标三维空间速度，以及速度大小
    
    - `ONE_D_RPM`: 用一个 `RPM ` 值控制所有电机
        - `target: np.ndarray[float64, shape=(1,)]`，目标转速
    
    - `ONE_D_PID`: 用一个 `PID` 值控制所有电机
        - `target: np.ndarray[float64, shape=(1,)]`，目标位置增量（`BaseRLAviary` 中仅高度）
    
- `gym` 环境的 `action` 是一个 `np.ndarray[float64, shape=(n,k,)]`
    - `action` 列表的第一维代表 `n` 个无人机
    - `action[n]` can be of length 1, 3, or 4, and represent RPMs, desired thrust and torques, or the next target position to reach using PID control.
        - $k=1$: `ActionType.ONE_D_RPM`, `ActionType.ONE_D_PID`
        - $k=3$: `ActionType.PID`
        - $k=4$: `ActionType.RPM`, `ActionType.VEL`
- `BaseRLAviary.py` 的 `_preprocessAction()` 将 `action` 翻译成四个舵机的 RPM 速度。
    - `RPM`: `rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))`
    - `PID`: 用 `target` 计算出下一步飞往的位置，由 `DSLPIDControl` 控制器的 `computeControl` 计算得出 `rpm[k,:]`
    - `VEL`: 用  `target` 计算出速度方向向量，进而得到不超过速度限制的目标速度向量，由 `DSLPIDControl` 控制器的 `computeControl` 在控制位置（pos）与偏航角（yaw）不变的情况下计算得出 `rpm[k,:]`
    - `ONE_D_RPM`: `rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)`
    - `ONE_D_PID`: 由 `DSLPIDControl` 控制器的 `computeControl` 计算得出 `rpm[k,:]`，此处的 `target` 仅用作高度增量 `target_pos=state[0:3]+0.1*np.array([0,0,target[0]]`

### 位置与方向

- `initial_xyzs`
    - 在没有传入参数时，`XYZ` 默认为 `[4 * i * L, 4 * i * L, COLLISION_H/2 - COLLISION_Z_OFFSET + 0.1]]`
        - `i` 是无人机编号
        - `L` 是模型的 `URDF_TREE[0].attrib['arm']`
        - `COLLISION_H` 是模型的碰撞高度：`URDF_TREE[1][2][1][0].attrib['length']`
        - `COLLISION_Z_OFFSET` 是模型在 Z 轴的碰撞体积偏移量

- `initial_rpys`
    - 在没有传入参数时，`RPY` 默认为 `[0, 0, 0]`，摄像头朝向为 `[1, 0, 0]`

- Action
    - 使用 `PID`、`VEL` 进行控制时不会更改摄像头视角

## 问题

1. 速度跟随应该得包含位置跟随？但二者似乎是分开控制的。需要设计流程。
2. 实际上代码中的做法是给定一个目标速度由 PID 去控制，那么我的 RL 应该怎么设计 action 呢？
3. 实现较为复杂，我是否应该自己重构一个环境？其中还包含人物的 3D 建模过程。
4. 我希望先去做一些关于控制（RL/PID）与识别（CV）调研。或者考虑帮博士干活，这个方向如果没有项目支撑，似乎过于难找切入点，RL 本身可能无竞争力（默罕默德不推荐）。
5. How to design the action? Should the drone follow the location & speed together? And how about the obstacle avoidance?
6. How to model people and let them run with random velocity in the env?
7. How to follow one persom within them? Give them different faces from real-world database?
8. 似乎没有实现 rpy 的直接调整