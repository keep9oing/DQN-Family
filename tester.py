import gym
import torch
import dqn
import ddqn
from torch.distributions import Categorical

import matplotlib.pyplot as plt


# Set device and load model
if torch.cuda.is_available(): # Use GPU
    device_num = torch.cuda.current_device()
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    print("Use GPU")
    print("current device is: ", torch.cuda.current_device())
    print("device name is: ",torch.cuda.get_device_name(device_num))   
else: # Use CPU
    device = torch.device('cpu')
    print("Use CPU")


# load environment
env = gym.make('MountainCar-v0')

model = dqn.Q_net(state_space=env.observation_space.shape[0], 
              action_space=env.action_space.n).to(device)
model.load_state_dict(torch.load('DQN.pth', map_location=device))

# Test setting
episodes = 1000
print_interval = 100
score = 0
score_sum = 0
epsilon = 0

for i_episode in range(episodes):
    # set initial state for environment
    state = env.reset()
    done = False

    while not done:
        # env.render()
        a = model.sample_action(torch.from_numpy(state).float().to(device), epsilon)
        s_prime, r, done, info = env.step(a)
        state = s_prime
        score += r
        score_sum += r

    if i_episode%print_interval==0 and i_episode!=0:
        print("n_episode :{}, score : {:.1f}, eps : {:.1f}%".format(
            i_episode, score/print_interval, epsilon*100))
        score = 0.0

print("Average Reward:", score_sum/episodes)

env.close()
