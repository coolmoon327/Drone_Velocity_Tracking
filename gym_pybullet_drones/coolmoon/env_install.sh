# URDF models: https://github.com/ChenEating716/pybullet-URDF-models

git clone https://github.com/ChenEating716/pybullet-URDF-models.git 
pip3 install -e pybullet-URDF-models/

# URDF human: https://github.com/robotology/human-gazebo?tab=readme-ov-file

git clone https://github.com/robotology/human-gazebo.git

# conda env

conda create --name pybullet python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tensorboard numpy scipy
pip install gymnasium stable-baselines3 pybullet transforms3d

# Run in background

conda activate pybullet
nohup python main.py >> log.txt &