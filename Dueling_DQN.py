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
action_num = 6
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

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=2,out_channels=32,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.fc1=nn.Linear(in_features=7*7*64,out_features=512)
        self.fc2=nn.Linear(512,128)
        #self.fc1 = nn.Linear(in_features=9*9*32, out_features=256)
        self.fc2_adv=nn.Linear(in_features=128,out_features=action_num)
        self.fc2_value = nn.Linear(in_features=128,out_features=1)
        #self.action_num=ACTION_NUM
    def forward(self,input):
        output=F.relu(self.conv1(input))
        output=F.relu(self.conv2(output))
        output=F.relu(self.conv3(output))
        output=output.view(output.size(0),-1)
        output=F.relu(self.fc1(output))
        output=F.relu(self.fc2(output))
        output_adv=self.fc2_adv(output)
        output_value = self.fc2_value(output)
        q_value = output_value + output_adv - output_adv.mean(1).reshape(-1,1)


        return q_value
       
class replay_memory():
    def __init__(self,capacity,min_delta = 1e-5):
        self.capacity = capacity
        self.min_delta = min_delta
        self.memory = deque(maxlen=capacity)
        self.deltas = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition',['state_cnn','action','reward','done'])

    def __len__(self):
        return len(self.memory)
    def add(self,state_cnn,action,reward,done):
        e = self.Transition(state_cnn,action,reward,done)
        self.memory.append(e)
        self.deltas.append(max(self.deltas) if len(self.deltas) > 0 else self.min_delta)

    def sample(self,batch_size,priority = 0.5):
        deltas = np.array(list(np.array(self.deltas))[:-num_agents])
        probs = deltas**priority / np.sum(deltas**priority)
        #rand_idx = np.random.randint(1,len(self.memory)-num_agents,batch_size)
        rand_idx = np.random.choice(range(len(self.memory)-num_agents), batch_size, p = probs,replace = True)
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
        probabilities = torch.from_numpy(np.vstack([probs[idx] for idx in rand_idx])).float()

        next_state_cnn = torch.cat((state_cnn, next_state_cnn), 1)
        state_cnn = torch.cat((prev_state_cnn, state_cnn), 1)
        return state_cnn.type(Tensor), next_state_cnn.type(Tensor), action, reward.type(Tensor), done.type(Tensor), probabilities, rand_idx
  
    def update_deltas(self,idxs,deltas):
        for i,idx in enumerate(idxs):
            self.deltas[idx] = deltas[i] + self.min_delta    

class Agent():
    def __init__(self):
        super(Agent,self).__init__()
        self.value_net, self.target_net = DQN().cuda(), DQN().cuda()
        self.step_counter = 0
        self.buffer = replay_memory(300000)
        self.batch_size = 32
        self.learn_every = 1
        self.global_update_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.value_net.parameters())
        self.criterion = nn.MSELoss()
        self.i_step = 0
        self.exploration_step = 0
        self.last_state_cnn = np.zeros((num_agents,1,84,84))
        self.last_action = np.zeros((num_agents, 1))
        self.loss_lst = []
        self.entropy_lst = []
        self.weight_importance = 0.5
        self.beta_increment_per_sampling = 0.001

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
    
    def step(self,state_cnn,action,reward,done, priority = 0.5):
        state_cnn = self.get_obs_cnn(state_cnn)
        for i in range(num_agents):
            self.buffer.add(state_cnn[i],action[i],reward[i],done)
        self.i_step = (self.i_step + 1)%self.learn_every
        if self.i_step == 0:
            if len(self.buffer) > self.batch_size:
                exp_batch = self.buffer.sample(self.batch_size,priority)
                self.learn(exp_batch)


    def select_action(self,obs,done):
        self.step_counter += 1
        a = 0.9
        b = (4000 - max(0, self.step_counter - 1000))
        threshold = 0.1 + max(0, a * b / 4000)
        print('Threshold:',threshold)
        print('Timestep:',self.step_counter)
        state = self.get_obs_cnn(obs).unsqueeze(1) #state: (num_agents, 84, 84)
        new_state_cnn = self.get_new_cnn(state)
        action_value = torch.zeros((num_agents, action_num))
        action_index = np.zeros((num_agents,),dtype = np.uint8)
        for i in range(num_agents):
            input = new_state_cnn[i]
            input_2 = torch.from_numpy(input).cuda()
            if random.random() <= threshold:
                action_index[i] = random.randint(0,5)
            else:
                action_value[i] = self.value_net.forward(input_2.unsqueeze(0).float())
                action_index[i] = action_value[i].max(0)[1].item()
        if done.item(0) != True:
            self.last_state_cnn = state
            self.last_action = action_index
        elif done.item(0) == True:
            self.last_state_cnn = np.zeros((num_agents,1, 84, 84))
            self.last_action = np.zeros((num_agents, 1))
        return action_index

    def learn(self,exp_batch):
        if self.step_counter < 1000:
            return
        state_cnn,next_state_cnn,action,reward,done,probs,exp_idxs = exp_batch
        q_values = self.value_net(state_cnn)
        next_q_values = self.value_net(next_state_cnn)
        next_q_state_values = self.target_net(next_state_cnn).detach()
        q_value = q_values.gather(1,action.long())
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values,1)[1].unsqueeze(1))

        q_target = reward +  (1 - done) * GAMMA * next_q_value
        td_error = q_target - q_value        
        new_deltas = torch.abs(td_error.detach().squeeze(1)).cpu().numpy()
        self.buffer.update_deltas(exp_idxs, new_deltas)
        self.optimizer.zero_grad()
        weights = (((1/len(self.buffer))*(1/probs))**self.weight_importance).cuda()
        #loss = (q_value - q_target).pow(2).mean() 
        loss = torch.mean((td_error**2)*weights)
        print('loss = ',loss)
        self.weight_importance = np.min([1., self.weight_importance  + self.beta_increment_per_sampling])
        loss.backward()
        self.optimizer.step()

        self.soft_updates()


    def soft_updates(self):
        tau = self.global_update_rate
        for value_params, target_params in zip(self.value_net.parameters(), self.target_net.parameters()):
            target_params.data.copy_(tau * value_params.data + (1-tau)*target_params.data)
