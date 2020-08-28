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

        self.Linear1 = nn.Linear(state_space, 64)
        self.Linear2 = nn.Linear(64, 64)
        self.Linear3 = nn.Linear(64, action_space)

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

    env_name = "LunarLander-v2"
    exp_num = '1'

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/'+env_name+'_DQN_'+exp_num)

    # Set parameters
    batch_size = 64
    learning_rate = 0.0005
    buffer_len = int(100000)
    min_buffer_len = 64
    episodes = 2000
    print_per_iter = 100
    target_update_period = 4
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    tau = 1e-3
    max_step = 2000

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
            replay_buffer.put((s, a, r/100.0, s_prime, done_mask))
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

        # Log the reward
        writer.add_scalar('Rewards per episodes', score, i)
        score = 0
            



    writer.close()
    env.close()

    def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)
    
    save_model(Q,'DQN_'+exp_num+'.pth')