# DQN-Family

Comparison DQN variations with pytorch. 
MountainCar-v0 is used for experiment.

# Requirement
* pytorch
* gym
* tensorboard
* numpy

# Mountain Car
![MountainCarGIF](./assets/gifs/MountainCar.gif)
![MountainCarReward](./assets/imgs/MountainCar.png)

Rewards per episodes training.

DQN: Blue

Double DQN: Pink

# Test result

test with 0 epsilon. 1k times average reward.

* DQN: -131.131
* Double DQN: -128.632

# TODO
- [x] DQN
- [x] Double DQN
- [ ] Dueling DQN
- [ ] Dueling Double DQN
