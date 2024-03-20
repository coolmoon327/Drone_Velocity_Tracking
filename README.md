# drone-velocity-tracking

## How to run

Our customized environment is availble at `./gym_pybullet_drones/coolmoon/`.

The start point is `./main.py`, you can use `python main.py` to run.

Follow the `./gym_pybullet_drones/coolmoon/env_install.txt` to build the Python env.

## Details about the customized environment

There are 2 types of Python files in `./gym_pybullet_drones/coolmoon/`:
- RL controller: `PPO_Visual.py`, `PPO_test.py`
- RL env wrapper: `Aviary_FrontVelocity.py`, `TestAvairy.py`

`PPO_test.py` and `TestAvairy.py` are only for environment testing,
while `PPO_Visual.py` and `Aviary_FrontVelocity.py` are used to test our proposed method.
