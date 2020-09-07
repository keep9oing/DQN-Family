import sys
from typing import Dict, List, Tuple

import gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


# Q_network
class Q_net(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None):
        super(Q_net, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.Feature_layer = nn.Linear(state_space, 64)
        self.Feature_value = nn.Linear(64, 32)
        self.Feature_advantage = nn.Linear(64, 32)
        self.Value_layer = nn.Linear(32, 1)
        self.Advantage_layer = nn.Linear(32, action_space)

    def forward(self, x):
        feature = F.relu(self.Feature_layer(x))
        value_feature = F.relu(self.Feature_value(feature))
        advantage_feature = F.relu(self.Feature_advantage(feature))
        value = F.relu(self.Value_layer(value_feature))
        advantage = F.relu(self.Advantage_layer(advantage_feature))
        return value + advantage - advantage.mean()

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0,1)
        else:
            return self.forward(obs).argmax().item()



# Replay buffer
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def put(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


def train(q_net=None, target_q_net=None, replay_buffer=None,
          device=None, 
          optimizer = None,
          batch_size=64,
          learning_rate=1e-3,
          gamma=0.99):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples = replay_buffer.sample()

    
    states = torch.FloatTensor(samples["obs"]).to(device)
    actions = torch.LongTensor(samples["acts"].reshape(-1,1)).to(device)
    rewards = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
    next_states = torch.FloatTensor(samples["next_obs"]).to(device)
    dones = torch.FloatTensor(samples["done"].reshape(-1,1)).to(device)

    # Define loss
    argmax_q_net = q_net(next_states).argmax(1).unsqueeze(1)
    q_target = target_q_net(next_states).detach().gather(1, argmax_q_net)
    target = rewards + gamma*q_target*dones

    q_out = q_net(states)
    q_a = q_out.gather(1, actions)
    loss = F.smooth_l1_loss(q_a, target)
    
    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)

if __name__ == "__main__":

    # Determine seeds
    model_name = "D3QN"
    env_name = "LunarLander-v2"
    seed = 1
    exp_num = 'SEED'+'_'+str(seed)

    # Set gym environment
    env = gym.make(env_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/'+env_name+"_"+model_name+"_"+exp_num)

    # Set parameters
    batch_size = 64
    learning_rate = 0.0005
    buffer_len = int(100000)
    min_buffer_len = batch_size
    episodes = 2000
    print_per_iter = 100
    target_update_period = 4
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    tau = 5*1e-3
    max_step = 2000
 
    # Create Q functions
    Q = Q_net(state_space=env.observation_space.shape[0], 
              action_space=env.action_space.n).to(device)
    Q_target = Q_net(state_space=env.observation_space.shape[0], 
                     action_space=env.action_space.n).to(device)

    Q_target.load_state_dict(Q.state_dict())

    # Create Replay buffer
    replay_buffer = ReplayBuffer(env.observation_space.shape[0],
                                            size=buffer_len, batch_size=batch_size)

    # Set optimizer
    score = 0
    score_sum = 0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start

    # Train
    for i in range(episodes):
        s = env.reset()
        done = False
        
        for t in range(max_step):
            # if i % print_per_iter == 0:
            #     env.render()

            # Get action
            a = Q.sample_action(torch.from_numpy(s).float().to(device), epsilon)

            # Do action
            s_prime, r, done, _ = env.step(a)
            # r += s_prime[0] ## For MountainCar

            # make data
            done_mask = 0.0 if done else 1.0
            replay_buffer.put(s, a, r/100.0, s_prime, done_mask)
            s = s_prime
            
            score += r
            score_sum += r


            if len(replay_buffer) >= min_buffer_len:
                train(Q, Q_target, replay_buffer, device, 
                        optimizer=optimizer,
                        batch_size=batch_size,
                        learning_rate=learning_rate)

                if (t+1) % target_update_period == 0:
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()):
                            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
                
            if done:
                break

        epsilon = max(eps_end, epsilon * eps_decay) #Linear annealing

        if i % print_per_iter == 0 and i!=0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            i, score_sum/print_per_iter, len(replay_buffer), epsilon*100))
            score_sum=0.0
            save_model(Q,'D3QN_'+exp_num+'.pth')

        # Log the reward
        writer.add_scalar('Rewards per episodes', score, i)
        score = 0

    writer.close()
    env.close()