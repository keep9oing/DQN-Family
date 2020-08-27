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

        self.Linear1 = nn.Linear(state_space, 128)
        self.Linear2 = nn.Linear(128, 128)
        self.Linear3 = nn.Linear(128, action_space)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        return self.Linear3(x)

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0,1)
        else:
            return self.forward(obs).argmax().item()


# Replay buffer
class Replay_buffer():
    def __init__(self, max_buffer_size=10000):
        self.max_buffer_size = max_buffer_size
        self.buffer = collections.deque(maxlen=self.max_buffer_size)

    def put(self, data):
        self.buffer.append(data)

    def sample(self, device=None, sample_size=32):
        batch = random.sample(self.buffer, k=sample_size)

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float, device=device), torch.tensor(a_lst, device=device), \
               torch.tensor(r_lst,dtype=torch.float, device=device), torch.tensor(s_prime_lst, dtype=torch.float, device=device), \
               torch.tensor(done_mask_lst, device=device)

    def __len__(self):
        return len(self.buffer)


def train(q_net=None, target_q_net=None, replay_buffer=None,
          device=None, 
          optimizer = None,
          batch_size=64,
          learning_rate=1e-3,
          epochs=1,
          gamma=0.99):

    assert device is not None, "None Device input: device should be selected."

    for i in range(epochs):
        # Get batch from replay buffer
        s, a, r, s_prime, done_mask = replay_buffer.sample(sample_size=batch_size, device=device)

        # Define loss
        q_target_max = target_q_net(s_prime).max(1)[0].unsqueeze(1).detach()
        target = r + gamma*q_target_max*done_mask
        q_out = q_net(s)
        q_a = q_out.gather(1, a)
        loss = F.smooth_l1_loss(q_a, target)

        # Update Network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":

    env_name = "MountainCar-v0"
    exp_num = '1'

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/'+env_name+'_DQN_'+exp_num)

    # Set parameters
    batch_size = 64
    learning_rate = 0.0001
    buffer_len = 100000
    min_buffer_len = 10000
    episodes = 60000
    print_per_iter = 500
    target_update_period = 30

    # Set gym environment
    env = gym.make(env_name)
 
    # Create Q functions
    Q = Q_net(state_space=env.observation_space.shape[0], 
              action_space=env.action_space.n).to(device)
    Q_target = Q_net(state_space=env.observation_space.shape[0], 
                     action_space=env.action_space.n).to(device)

    Q_target.load_state_dict(Q.state_dict())

    # Create Replay buffer
    replay_buffer = Replay_buffer(max_buffer_size=buffer_len)

    # Set optimizer
    score = 0
    score_sum = 0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    # Train
    for i in range(episodes):
        s = env.reset()
        done = False
        epsilon = max(0.001, 1 - 0.0001*i) #Linear annealing 

        while not done:
            # if i % print_per_iter == 0:
            #     env.render()

            # Get action
            a = Q.sample_action(torch.from_numpy(s).float().to(device), epsilon)

            # Do action
            s_prime, r, done, _ = env.step(a)
            # print(s_prime[0])
            r = r+s_prime[0]

            # make data
            done_mask = 0.0 if done else 1.0
            replay_buffer.put((s, a, r/100.0, s_prime, done_mask))
            s = s_prime
            
            score += r
            score_sum += r
            if done:
                break

        if len(replay_buffer) >= min_buffer_len:
            train(Q, Q_target, replay_buffer, device, 
                    optimizer=optimizer,
                    batch_size=batch_size,
                    learning_rate=learning_rate)

        if i % print_per_iter == 0 and i!=0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            i, score_sum/print_per_iter, len(replay_buffer), epsilon*100))
            score_sum=0.0

        # Log the reward
        writer.add_scalar('Rewards per episodes', 
                            score, i)
        score = 0
            

        if i % target_update_period == 0:
            Q_target.load_state_dict(Q.state_dict())

    writer.close()
    env.close()

    def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)
    
    save_model(Q,'DQN_'+exp_num+'.pth')