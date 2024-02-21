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
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.coolmoon.TestAviary import TestAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 24

DEFAULT_OBS = ObservationType('rgb') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('vel') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1

def evaluate(filename):
    input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

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
                                                pyb_freq=DEFAULT_SIMULATION_FREQ_HZ, ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,),
                                n_envs=1,
                                seed=0
                                )
    eval_env = TestAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, pyb_freq=DEFAULT_SIMULATION_FREQ_HZ, ctrl_freq=DEFAULT_CONTROL_FREQ_HZ)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    model = PPO('MlpPolicy',
                train_env,
                # tensorboard_log=filename+'/tb/',
                verbose=1)
                # , device = "mps")

    #### Target cumulative rewards (problem-dependent) ##########
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 949.5
    else:
        target_reward = 920.
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
    model.learn(total_timesteps=int(1e7),
                callback=eval_callback,
                log_interval=100)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

def run():
    # filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    # if not os.path.exists(filename):
    #     os.makedirs(filename+'/')
    # train(filename)
    
    filename = os.path.join(DEFAULT_OUTPUT_FOLDER, "save-02.17.2024_18.39.15")
    evaluate(filename)


def test():
    # filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    # if not os.path.exists(filename):
    #     os.makedirs(filename+'/')

    env = TestAviary(gui=DEFAULT_GUI,
                        obs=DEFAULT_OBS,
                        act=DEFAULT_ACT,
                        pyb_freq=DEFAULT_SIMULATION_FREQ_HZ,
                        ctrl_freq=DEFAULT_CONTROL_FREQ_HZ,
                        record=DEFAULT_RECORD_VIDEO,
                        )

    print('[INFO] Action space:', env.action_space)
    print('[INFO] Observation space:', env.observation_space)

    obs, info = env.reset(seed=88, options={})
    action = np.zeros((1, 4))
    # action = np.zeros((1, 3))
    start = time.time()
    for i in range((env.EPISODE_LEN_SEC+2)*env.CTRL_FREQ):
    # for i in range(10):
        action = np.array([[-1,0,0,1]])
        # action = np.array([[-0.5,0,0.1]])
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            obs = env.reset(seed=42, options={})

    env.close()



if __name__ == '__main__':
    run()
    # test()
