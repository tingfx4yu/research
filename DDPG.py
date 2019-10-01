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
GAMMA = 0.99
action_num = 8
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
def ob_process(img):
    img=torch.FloatTensor(torch.from_numpy(img/255))
    #img=img.unsqueeze(0)
    return img

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim,lim)

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

class Actor(nn.Module):
    def __init__(self):
        super(Actor,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1=nn.Conv2d(in_channels=2,out_channels=32,kernel_size=7,stride=4)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.fc1=nn.Linear(in_features=64*7*7,out_features=64)
        self.fc2=nn.Linear(in_features=64,out_features=action_num)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
    def forward(self,input):
        output=F.relu(self.conv1(input))
        #output = self.bn1(output)
        output=F.relu(self.conv2(output))
        #output = self.bn2(output)
        output=F.relu(self.conv3(output))
        output=output.view(output.size(0),-1)
        output=F.relu(self.fc1(output))
        #output=F.relu(self.fc2(output))
        output=self.fc2(output)
        action_prob = F.softmax(output,dim = 1)

        return action_prob
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1=nn.Conv2d(in_channels=2,out_channels=32,kernel_size=7,stride=4)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.fc1=nn.Linear(in_features=64*7*7,out_features=64)
        self.fc2=nn.Linear(in_features=64,out_features=1)
    def forward(self,input):
        output=F.relu(self.conv1(input))
        output=F.relu(self.conv2(output))
        output=F.relu(self.conv3(output))
        output=output.view(output.size(0),-1)
        output=F.relu(self.fc1(output))
        output=self.fc2(output)
        
        return output
       