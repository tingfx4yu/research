import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple,deque
import random
import numpy as np
import matplotlib.pyplot as plt

num_agents = 30
action_num = 8
GAMMA = 0.99
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def ob_process(img):
    img=torch.FloatTensor(torch.from_numpy(img/255))
    #img=img.unsqueeze(0)
    return img

class Actor(nn.Module):
    def __init__(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(2,32,8,4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,action_num)
        )
    def forward(self,input):
        output = self.conv1(input)
        output = output.view(output.size(0),-1)
        output = self.fc1(output)
        action_prob = F.softmax(output,1)
        return action_prob

class Critic(nn.Module):
    def __init__(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(2,32,8,4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self,input):
        output = self.conv1(input)
        output = output.view(output.size(0),-1)
        value = self.fc1(output)
        return value

class replay_memory:
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = deque(maxlen = capacity)
        #self.memory = []
        self.Transition = namedtuple('Transition',['state_cnn','action','reward','done'])
    def __len__(self):
        return len(self.memory)
    def add(self,state_cnn,action,reward,done):
        e = self.Transition(state_cnn,action,reward,done)
        self.memory.append(e)
    def sample(self,batch_size):
        rand_idx = np.random.randint(1,len(self.memory)-num_agents,batch_size)
        next_rand_idx = rand_idx + num_agents
        prev_rand_idx = rand_idx - num_agents
        state = [self.memory[i] for i in rand_idx]
        next_state = [self.memory[i] for i in next_rand_idx]
        prev_state = [self.memory[i] for i in prev_rand_idx]

        prev_state_cnn = torch.from_numpy(np.vstack([e.state_cnn for e in prev_state if e is not None])).float().view(batch_size,1,84,84)
        state_cnn = torch.cat([e.state_cnn for e in state if e is not None]).view(batch_size,1,84,84)
        next_state_cnn = torch.cat([e.state_cnn for e in next_state if e is not None]).view(batch_size,1,84,84)

        action = torch.from_numpy(np.vstack([e.action for e in state if e is not None])).float().cuda()
        reward = torch.from_numpy(np.vstack([e.reward for e in state if e is not None])).float()
        done = torch.from_numpy(np.vstack([e.done for e in state if e is not None]).astype(np.uint8)).float()

        next_state_cnn = torch.cat((state_cnn, next_state_cnn), 1)
        state_cnn = torch.cat((prev_state_cnn, state_cnn), 1)
        return state_cnn.type(Tensor), next_state_cnn.type(Tensor), action, reward.type(Tensor), done.type(Tensor)
    
class Agent():
    def __init__(self):
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.clip = 0.2
        self.grad_norm = 0.5
        self.ppo_update_time = 10
        self.buffer = replay_memory(320000)
        self.batch_size = 512
        self.step_counter = 0
        self.actor_opt = optim.Adam(self.actor_net.parameters(),1e-3)
        self.critic_opt = optim.Adam(self.critic_net.parameters(),3e-3)

    def get_obs_cnn(self,obs):
        temp = []
        for i in range(num_agents):
            temp.append(np.r_[obs["image"][i]])
        temp = np.r_[temp]
        t = ob_process(temp)
        return t

    def get_new_cnn(self, t):
        t = np.concatenate((self.last_state_cnn, t), axis=1)
        return t
    
    def select_action(self,obs):
        state = self.get_obs_cnn(obs).unsqueeze(1)
        new_state_cnn = self.get_new_cnn(state)
        action_index = 


    
            
