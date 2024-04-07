# 更新内容

## 2024.2.17

1. 在 `PPO_test.py` 新建 `test` 函数用于前期测试，发现拍照的时间频率必须对上控制的频率（有一定的倍数关系），于是手动设置 `pyb_freq` 和 `ctrl_freq`
2. 为环境 `TestAviary.py` `BaseRLAviary.py` 添加 `output_folder`
    - 实际上并未使用，依旧使用 `BaseAviary.py`  的默认路径进行 record
3. 修改了从 `BaseRLAviary.py` 调用 `_exportImage` 时存在的路径 bug: 没有在子目录前加 `/`
4. 修改了 `BaseAviary.py` 中 `_startVideoRecording` 新建路径的 bug: 使用 `os.path.dirname` 提取路径名，但路径字符串最后不是 `/`，导致忽视了子路径名
5. 在 `BaseAviary.py` 统一了视频与图片的记录目录
    - 但是视频录制似乎有问题，暂时弃用视频部分

## 2024.2.24

1. 不用 obstacle 来放 target，而是在环境中添加 target 控制对应的函数
2. 修改 velocity 的控制内容，决定不再改变垂直位置上的速度分量
3. 为无人机掉落等添加惩罚
4. 加入对相对速度的判定，并设置对应的终止状态与 reward
5. 完成第一版 cnn ppo 的搭建

## 2024.2.28

1. 修改 `BaseAviary.py` 里图片的分辨率和采样率

## 2024.3.3

1. 增加环境，只测试向前的速度跟随
2. 考虑将环境控制频率与RL控制频率分离

## 2024.3.10

1. 使用 obj 模型作为 target
2. obj 加载后，坐标系有点问题，包括它的速度坐标都需要转换
3. 需要验证 action 的速度是否正确（研究 computeControl），并进一步设计 reward

## 2024.3.11

1. 修改速度设计
2. 修改图片特征提取位置
3. 修改了 step 完毕才重置 has_target 的 bug
4. 修改 truncated 的惩罚过程
5. BaseAviary 的 step 改成最后才计算 reward

## 2024.3.13 TODO

1. 可以把图片不全的情况直接维持速度，并且 truncate 改成连续三次没有看见人才 G
2. reward 归一化
3. 无人机调高，不然一飞就丢人
4. 取消 y 轴 center

为什么 target_pos 和现在的一样？？？

大问题：速度太快，倾斜过大，丢失目标

## 2024.3.14

1. 包括目标的速度在内全部调小 10 倍，把无人机调低到用户一个平面

## 2024.3.16

1. 试着用更多的动力学信息作为 state，效果也不佳

## 2024.4.5

1. 换了模型
2. 调大 FOV
3. 完全重制 Test，先试试用 yolo 的输出来进行位置到达

## 2024.4.7

1. 直接把目标改为目标宽度的大小占比为 27%，并随之修改 reward