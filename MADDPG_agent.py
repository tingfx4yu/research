import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.optim import Adam


#-------hyperParameter-----------

num_agents = 10
batch_size = 32
dim_obs = 8
dim_act = 2
capacity = 10000
explore_step = 5000
GAMMA = 0.99
tau = 0.01
scale_reward = 0.01
use_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

#----------Basic Function-----------

def ob_process(img):
    img=torch.FloatTensor(torch.from_numpy(img/255))
    #img=img.unsqueeze(0)
    return img

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)

def get_obs_cnn(obs):
    temp = []
    for i in range(num_agents):
        temp.append(np.r_[obs["image"][i]])
    temp = np.r_[temp]
    t = ob_process(temp)
    return t

def get_obs_ir(obs):
    temp = []
    for i in range(num_agents):
        temp.append(np.r_[obs["ir"][i]])
    temp = np.r_[temp]
    #print(temp)
    return temp



def act2lst(act_array):
    action_lst = []
    act_array = list(act_array)
    #length = len(act_array)
    for i in act_array:
        if i >= 0 and i < 0.333:
            k = 0
        elif i>=0.333 and i < 0.6666:
            k = 0.5
        elif i >= 0.6666 and i <= 1:
            k = 1
        action_lst.append(k)
    if action_lst == [0,0]:
        return 0
    elif action_lst == [0,0.5]:
        return 1
    elif action_lst == [0,1]:
        return 2    
    elif action_lst ==[0.5,0]:
        return 3
    elif action_lst == [0.5,0.5]:
        return 4
    elif action_lst == [0.5,1]:
        return 5
    elif action_lst == [1,0]:
        return 6
    elif action_lst == [1,0.5]:
        return 7
    elif action_lst == [1,1]:
        return 8



# -----------Net Structure---------------

class Critic(nn.Module):
    def __init__(self,num_agents,dim_o,dim_a):
        super(Critic,self).__init__()
        self.num_agents = num_agents
        self.dim_o = dim_o
        self.dim_a = dim_a
        obs_dim = dim_o * num_agents
        act_dim = dim_a * num_agents

        self.fc1 = nn.Linear(obs_dim,1024)
        self.fc2 = nn.Linear(1024+act_dim,512)
        self.fc3 = nn.Linear(512,300)
        self.fc4 = nn.Linear(300,1)
    def forward(self,input,acts):
        output = F.relu(self.fc1(input))
        output = torch.cat([output,acts],1)
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = self.fc4(output)
        return output

class Actor(nn.Module):
    def __init__(self):
        super(Actor,self).__init__()
        self.conv1=nn.Conv2d(in_channels=2,out_channels=32,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        self.fc1 = nn.Linear(in_features=7*7*64,out_features = 128)
        self.fc2=nn.Linear(in_features=128,out_features=2)

    def forward(self,input):
        output=F.relu(self.conv1(input))
        output=F.relu(self.conv2(output))
        output=F.relu(self.conv3(output))
        output=output.view(output.size(0),-1)
        output=F.relu(self.fc1(output))
        output=self.fc2(output)
        return output

#--------------------Buffer------------------------

class replay_memory:
    def __init__(self):
        self.memory = deque(maxlen = capacity)
        self.Transition = namedtuple('Transition',['state_cnn','state_ir','action','reward','done'])
    def __len__(self):
        return len(self.memory)
    def add(self,state_cnn,state_ir,action,reward,done):
        e = self.Transition(state_cnn,state_ir,action,reward,done)
        self.memory.append(e)
    def sample(self,batch_size):
        rand_idx = np.random.randint(1,len(self.memory)-1,batch_size)
        next_rand_idx = rand_idx + 1
        prev_rand_idx = rand_idx - 1
        state = [self.memory[i] for i in rand_idx]
        next_state = [self.memory[i] for i in next_rand_idx]
        prev_state = [self.memory[i] for i in prev_rand_idx]

        prev_state_cnn = np.vstack([e.state_cnn for e in prev_state if e is not None]).reshape(batch_size,num_agents,1,84,84)
        pre_state_ir = np.vstack([e.state_ir for e in prev_state if e is not None])

        state_cnn = np.concatenate([e.state_cnn for e in state if e is not None]).reshape(batch_size,num_agents,1,84,84)
        state_ir = np.concatenate([e.state_ir for e in state if e is not None]).reshape(batch_size,num_agents,-1)

        next_state_cnn = np.concatenate([e.state_cnn for e in next_state if e is not None]).reshape(batch_size,num_agents,1,84,84)
        next_state_ir =  np.concatenate([e.state_ir for e in next_state if e is not None]).reshape(batch_size,num_agents,-1)

        #action = torch.from_numpy(np.vstack([e.action for e in state if e is not None])).float()
        action = np.concatenate([e.action for e in state if e is not None]).reshape(batch_size,num_agents,2)
        reward = np.vstack([e.reward for e in state if e is not None])
        done = np.vstack([e.done for e in state if e is not None]).astype(np.uint8)

        next_state_cnn = np.concatenate((state_cnn, next_state_cnn), 2)
        state_cnn = np.concatenate((prev_state_cnn, state_cnn), 2)
        return state_cnn, next_state_cnn,state_ir,next_state_ir, action, reward, done
  
#--------------------MADDPG------------------------

class MADDPG():
    def __init__(self):
        self.actors = [Actor() for i in range(num_agents)]
        self.critics = [Critic(num_agents,dim_obs,dim_act) for i in range(num_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        self.last_state_cnn = np.zeros((num_agents,1,84,84))
        self.last_state_ir = np.zeros((num_agents,8))
        self.step = 0
        self.buffer = replay_memory()
        self.var = [1 for i in range(num_agents)]
        #self.random_number = [random.uniform(-0.5,0.5) for i in range(num_agents)]
        self.critic_optimizer = [Adam(x.parameters(),lr=0) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),lr=0) for x in self.actors]

        if torch.cuda.is_available():
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()


    def get_new_cnn(self,t):
        t = np.concatenate((self.last_state_cnn, t), axis=1)
        return t

    def learn(self):
        if self.step <= explore_step:
            return
        ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        for agent in range(num_agents):
            state_cnn,next_state_cnn,state_ir,next_state_ir,action,reward,done = self.buffer.sample(batch_size)
            state_cnn = torch.from_numpy(state_cnn).type(Tensor)
            next_state_cnn = torch.from_numpy(next_state_cnn).type(Tensor)
            state_ir = torch.from_numpy(state_ir).type(Tensor)
            next_state_ir = torch.from_numpy(next_state_ir).type(Tensor)
            action = torch.from_numpy(action).type(Tensor)
            reward = torch.from_numpy(reward).type(Tensor)
            non_final_mask = ByteTensor(list(map(lambda s: s is not None, next_state_cnn)))
            non_final_next_states = torch.stack([s for s in next_state_cnn if s is not None]).type(FloatTensor)
            non_final_next_states_ir = torch.stack([s for s in next_state_ir if s is not None]).type(FloatTensor)
            #whole_state = state_cnn.view(batch_size, -1)
            whole_state = state_ir.view(batch_size,-1)
            whole_action = action.view(batch_size,-1).type(Tensor)

            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state,whole_action)
            next_action = [self.actors_target[i](non_final_next_states[:,i,:,:,:]) for i in range(num_agents)]

            next_action = torch.stack(next_action)
            next_action = (next_action.transpose(0,1).contiguous())

            target_Q = torch.zeros(batch_size).type(Tensor)
            target_Q[non_final_mask] = self.critics_target[agent](non_final_next_states_ir.view(-1, num_agents * dim_obs),next_action.view(-1, num_agents * dim_act)).squeeze()

            target_Q = (target_Q.unsqueeze(1) * GAMMA) + (reward[:,agent].unsqueeze(1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q,target_Q.detach())
            #print(loss_Q)
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_cnn[:, agent, :,:,:]
            action_i = self.actors[agent](state_i)
            ac = action.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(batch_size, -1).type(Tensor)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            #print(actor_loss)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            #c_loss.append(loss_Q)
            #a_loss.append(actor_loss)
        
        if self.step % 100 == 0:
            for i in range(num_agents):
                soft_update(self.critics_target[i],self.critics[i],tau)
                soft_update(self.actors_target[i],self.actors[i],tau)

    def select_action(self,obs,done):
        actions = np.zeros((num_agents,dim_act))
        #try_l = np.zeros((num_agents*dim_act,))
        obs = get_obs_cnn(obs).unsqueeze(1)
        new_state_cnn = self.get_new_cnn(obs)
        act_lst = []

        for i in range(num_agents):   
            sb = torch.from_numpy(new_state_cnn[i,:]).type(Tensor).unsqueeze(0)
            #print(sb)
            act = self.actors[i].forward(sb).squeeze()

            act += torch.from_numpy(np.random.randn(2) *self.var[i]).type(Tensor)

            if self.step > explore_step and self.var[i] > 0.05:
                self.var[i] *= 0.99
            
            act = torch.clamp(act, 0,1)
            act_np = act.detach().cpu().numpy()
            #act_np = np.round(act_np,1)
            actions[i,:] = act_np
            action_number = act2lst(act_np)
            act_lst.append(action_number)
        if done.item(0) != True:
            self.last_state_cnn = obs 
            self.last_action = actions 
        elif done.item(0) == True:
            self.last_state_cnn = np.zeros((num_agents,1, 84, 84))
            self.last_action = np.zeros((num_agents, dim_act))
        self.step += 1 
        print('step:',self.step)
        return actions,act_lst


    def store_experience(self,obs,action,reward,done):
        state_cnn = get_obs_cnn(obs)
        state_ir = get_obs_ir(obs)
        self.buffer.add(state_cnn,state_ir,action,reward,done)




        





    






