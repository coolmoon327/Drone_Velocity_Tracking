"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from .TestAviary import TestAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 24

DEFAULT_OBS = ObservationType('rgb') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('vel') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.BatchNorm1d(n_flatten), 
                                    nn.Linear(n_flatten, 512), 
                                    nn.BatchNorm1d(512), 
                                    nn.ReLU(),
                                    nn.Linear(512, features_dim))

    def forward(self, observations):
        return self.linear(self.cnn(observations))

def evaluate(filename):
    input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path, device = "cuda:3")

    #### Show (and record a video of) the model's performance ##
    test_env = TestAviary(gui=DEFAULT_GUI,
                            obs=DEFAULT_OBS,
                            act=DEFAULT_ACT,
                            pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                            ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                            record=DEFAULT_RECORD_VIDEO)
    test_env_nogui = TestAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, pyb_freq=DEFAULT_SIMULATION_FREQ_HZ, ctrl_freq=DEFAULT_CONTROL_FREQ_HZ)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=1,
                output_folder=DEFAULT_OUTPUT_FOLDER
                )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            logger.log(drone=0,
                timestamp=i/test_env.CTRL_FREQ,
                state=np.hstack([obs2[0:3],
                                    np.zeros(4),
                                    obs2[3:15],
                                    act2
                                    ]),
                control=np.zeros(12)
                )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    # if DEFAULT_OBS == ObservationType.KIN:
    #     logger.plot()

def train(filename):
    train_env = make_vec_env(TestAviary,
                            env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, 
                                            pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                                            ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,),
                            n_envs=16,
                            vec_env_cls=SubprocVecEnv
                            )
    eval_env = TestAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, pyb_freq=DEFAULT_SIMULATION_FREQ_HZ, ctrl_freq=DEFAULT_CONTROL_FREQ_HZ)
    eval_env = Monitor(eval_env)

    #### Check the environment ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    # model = PPO('CnnPolicy', train_env, verbose=1, 
    model = PPO('MlpPolicy', train_env, verbose=1, 
                tensorboard_log=filename+'/tb/',
                # policy_kwargs={'features_extractor_class': CustomCNN}, 
                device = "cuda:3",)
                # batch_size=128,
                # n_steps = 500)
    
    #### Target cumulative rewards (problem-dependent) ##########
    target_reward = 1e6

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(1e8),
                callback=eval_callback,
                log_interval=100)

    eval_env.close()

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

def run(eval=False):
    if not eval:
        filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(filename):
            os.makedirs(filename+'/')
        train(filename)
    else:
        filename = os.path.join(DEFAULT_OUTPUT_FOLDER, "save-for-evaluate")
        evaluate(filename)


def test_velocity():
    env = TestAviary(obs=ObservationType('rgb'),
                    act=ActionType('vel'),
                    pyb_freq=240,
                    ctrl_freq=24,
                    gui=False,
                    record=False,
                    )

    print('=========================')
    print('[INFO] Action space:', env.action_space)
    print('[INFO] Observation space:', env.observation_space)
    print('=========================')

    obs, info = env.reset(seed=88, options={})
    # action = np.zeros((1, 4))
    action = np.array([[.1]])
    # action = np.array([[0.]])

    start = time.time()
    for i in range(1000):
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated or truncated:
            obs = env.reset(seed=42, options={})

    env.close()


def test():
    # filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    # if not os.path.exists(filename):
    #     os.makedirs(filename+'/')

    test_velocity()


if __name__ == '__main__':
    run()
    # test()
