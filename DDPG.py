import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple,deque
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal, Categorical

num_agents = 30
GAMMA = 0.99
action_num = 2
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
        #self.seed = torch.manual_seed(seed)
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
        output=F.relu(self.conv2(output))
        #output = self.bn2(output)
        output=F.relu(self.conv3(output))
        output=output.view(output.size(0),-1)
        output=F.relu(self.fc1(output))
        #output=F.relu(self.fc2(output))
        output=self.fc2(output)
        return output
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        #self.seed = torch.manual_seed(seed)
        self.conv1=nn.Conv2d(in_channels=2,out_channels=32,kernel_size=7,stride=4)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.fc1=nn.Linear(in_features=64*7*7,out_features=64)
        self.fc2= nn.Linear(64+8,64)
        self.fc2=nn.Linear(in_features=64,out_features=1)
    def forward(self,input,action):
        output=F.relu(self.conv1(input))
        output=F.relu(self.conv2(output))
        output=F.relu(self.conv3(output))
        output=output.view(output.size(0),-1)
        output=F.relu(self.fc1(output))
        output = torch.cat((output,action),dim = 1)
        output=self.fc2(output)
        
        return output

class Agent():
    def __init__(self):
        super(Agent,self).__init__()
        self.seed = random.seed()
        self.buffer = replay_memory(350000)
        self.step_counter = 0
        self.actor_local = Actor().cuda()
        self.actor_traget = Actor().cuda()
        self.Critic_local = Critic().cuda()
        self.Critic_target = Critic().cuda()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = 1e-4)
        self.critic_optimizer = optim.Adam(self.Critic_local.parameters(), lr = 3e-4)
        self.batch_size = 256
        self.random_walk = 1000
        self.decay_walk = 1500
        self.last_state_cnn = np.zeros((num_agents,1,84,84))
        self.last_action = np.zeros((num_agents, 1))
        self.noise = OUNoise(action_size, random_seed)

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

    def select_action(self,obs,done):
        self.step_counter += 1
        a = 0.9
        b = (self.decay_walk - max(0, self.step_counter - self.random_walk))
        threshold = 0.1 + max(0, a * b / self.decay_walk)
        print('Threshold:',threshold)
        print('Timestep:',self.step_counter)
        state = self.get_obs_cnn(obs).unsqueeze(1) #state: (num_agents, 84, 84)
        new_state_cnn = self.get_new_cnn(state)
        action_index = np.zeros((num_agents,),dtype = np.uint8)
        for i in range(num_agents):
            input = new_state_cnn[i]
            input_2 = torch.from_numpy(input).cuda()
            if random.random() <= threshold:
                action_index[i] = random.randint(0,7)
            else:
                action = self.actor_local(input_2).detach()
                action += self.noise
                action_index[i] = action.cpu().numpy()
        if done.item(0) != True:
            self.last_state_cnn = state
            self.last_action = action_index
        elif done.item(0) == True:
            self.last_state_cnn = np.zeros((num_agents,1, 84, 84))
            self.last_action = np.zeros((num_agents, 1))
        return action_index

    def store_experience(self,obs,action,reward,done):
        state_cnn = self.get_obs_cnn(obs)
        for i in range(num_agents):
            self.buffer.add(state_cnn[i],action[i],reward[i],done)

    def learn(self):
        if self.step_counter < self.random_walk:
            return
        state_cnn,next_state_cnn,actions,reward,done = self.buffer.sample(self.batch_size)
        action_next = self.actor_traget(next_state_cnn)
        Q_target_next = self.Critic_target(next_state_cnn,action_next)
        Q_targets = reward + (GAMMA * Q_target_next * (1-done))
        Q_expected = self.Critic_local(state_cnn,actions)
        critic_loss = F.mse_loss(Q_expected,Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        actions_pred = self.actor_local(state_cnn)
        actor_loss = - self.Critic_local(state_cnn,actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.Critic_local, self.Critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


        
        
    

        







