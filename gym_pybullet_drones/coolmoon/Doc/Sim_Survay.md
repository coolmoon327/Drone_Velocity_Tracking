# Brief Survay about Sim Envs



## General Comparison

> AirSim project was **archived** by Microsoft, and new machines (like my MBP with M3Max) may encounter difficulties that cannot be solved officially.

|                         | PyBullet                                                     | Gazebo                                                       | AirSim                                                       |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Platform**            | Windows, Linux, Mac OS                                       | Linux, Mac OS                                                | Windows, Linux                                               |
| **Cross-platform**      | Native                                                       | Gazebo on Windows is not fully functional.                   | AirSim has Docker image.  Mac OS (M3Max) meets problems when building UE4. |
| **Physics Engine**      | Bullet                                                       | ODE (Default), Bullet, Simbody, DART                         | PhysX                                                        |
| **Software Dependency** | None                                                         | Conda (On Windows)                                           | Unreal/Unity (Epic Game Launcher)                            |
| **Control**             | Code/API                                                     | Code/API, User Inputs                                        | Code/API, User Inputs                                        |
| **UI Quality**          | Low                                                          | High                                                         | High                                                         |
| **UI FPS**              | High                                                         | Medium                                                       | High (On good GPU)                                           |
| **UI Fidelity**         | Low                                                          | Medium                                                       | High                                                         |
| **Sensors**             | Limited (Camera, Motors' states)                             | Wide (Camera, Lidar, GPS, etc)                               | Wide (Camera, Depth Camera, Lidar, GPS, etc)                 |
| **Deployment**          | Easy to use                                                  | Work with other components (ROS, Ardupilot, Ardupilot_gazebo, etc.) | High hardware requirement                                    |
| **Model File**          | URDF (Unified Robot Description Format) & SDF (Simulation Description Format) | URDF & SDF                                                   | URDF & SDF & Unreal/Unity files                              |
| **Gym Support**         | Well supported                                               | 3rd-party Supported                                          | In active development (said by MS, but the project has been archived for 2 years) |
| **RL Drone Envs**       | [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) | [drl_uav](https://github.com/PX4-Gazebo-Simulation/drl_uav)  | [Official Guide: RL with Quadrotor](https://microsoft.github.io/AirSim/reinforcement_learning/) |
| **Simulation Speed**    | gym-pybullet-drones could generate 80× the data of the elapsed time | Depends on the codes                                         | Real-time                                                    |
| **Advances**            | Cross-platform, gym-supported, fast-simulation               | Functional                                                   | Functional, high-quality                                     |
| **Drawbacks**           | Low-quality, limited-sensors                                 | Low-quality, complicated                                     | High-hardware-requirement, real-time-simulation              |



## Regarding Human-speed Tracking

|                                         | PyBullet | Gazebo | AirSim |
| --------------------------------------- | -------- | ------ | ------ |
| **Linux Server (Deploy)**               | 😄        | 😄      | 😄      |
| **Macbook (Develop)**                   | 😄        | 😄      | 😈      |
| **Camera Images (Agent's Observation)** | 😄        | 😄      | 😄      |
| **Image Fidelity**                      | 😈        | 😓      | 😄      |
| **Control API (Agent's Action)**        | 😄        | 😄      | 😄      |
| **Env Wrapping for RL**                 | 😄        | 😈      | 😓      |

- When training, we just use the env without rendering. So the UI has no effect.
- All the three tools support **URDF** files, we can model objects in such forms in case trying to use another tool in the future.


