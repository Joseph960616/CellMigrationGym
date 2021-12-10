import numpy as np
import pybullet

#RL related package
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

STATE_CELL_LIST = ['ABarpppap', 'ABarppppa', 'ABarppppp', 'Caaaa', 'ABprapapp', 'Epra', 'ABprapaaa', \
				'ABprapaap', 'Cpaap', 'ABprapapa', 'ABarppapp', 'Caaap', 'Eprp', 'ABarpppaa', 'Eplp', \
				'ABarppapa', 'Epla', 'ABarppaap']

ACTIONS = [0,1,2,3,4,5,6,7]
N_STATES = (len(STATE_CELL_LIST) + 1) * 3
N_ACTIONS = len(ACTIONS)

#High level network
###input state should be all neighbor from the embryo at AI cell migrating period
SUBGOAL_LIST = ['ABarppppa','ABarppapp','ABarpppap','ABarppapa','ABarppaap','ABarpppaa']
N_ACTIONS_H = len(SUBGOAL_LIST) * (len(SUBGOAL_LIST) - 1)


# Hyper Parameters
BATCH_SIZE = 64
LR = 0.0001                   # learning rate
EPSILON = 0.3               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 1000   # target update frequency
MEMORY_CAPACITY = 8000

# BATCH_SIZE = 32
# LR = 0.0001                     # learning rate
# EPSILON = 0.8                   # greedy policy
# GAMMA = 0.98                    # reward discount
# TARGET_REPLACE_ITER = 1000      # target update frequency
# MEMORY_CAPACITY = 4000

N_CHANNEL = 2
N_INPUT = 1
INPUT_SIZE = 64
N_STATES_CNN = INPUT_SIZE * INPUT_SIZE * N_CHANNEL * N_INPUT



if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

np.random.seed(2)
torch.manual_seed(2)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 512)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(512, 1024)
        self.fc2.weight.data.normal_(0, 0.1)   # initialization 
        self.fc3 = nn.Linear(1024, 1024)
        self.fc3.weight.data.normal_(0, 0.1)   # initialization                
        self.out = nn.Linear(1024, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        # x = self.bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class CNN_Net(nn.Module):
    def __init__(self, ):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(N_CHANNEL * N_INPUT, 32, 5, 4, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )                               #output (32x16x16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )                               #output (64x8x8)
        self.out = nn.Linear(64*8*8, N_ACTIONS)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        if use_cuda:
            self.eval_net, self.target_net = Net().cuda(), Net().cuda()
        else:
            self.eval_net, self.target_net = Net(), Net()
        self.e_greedy = EPSILON
        self.learning_rate = LR
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        
        if use_cuda:
            x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0).cuda())
        else:
            x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # input only one sample
        if np.random.uniform() < self.e_greedy:   # greedy
            actions_value = self.eval_net.forward(x)
            if use_cuda:
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            else:
                action = torch.max(actions_value, 1)[1].data.numpy()[0]     # return the argmax

        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('Parameters updated')
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        if use_cuda:
            b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]).cuda())
            b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).cuda())
            b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).cuda())
            b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]).cuda())
        else:
            
            b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
            b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
            b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
            b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.cpu().numpy()


class DQN_CNN(object):
    def __init__(self):
        if use_cuda:
            self.eval_net, self.target_net = CNN_Net().cuda(), CNN_Net().cuda()
        else:
            self.eval_net, self.target_net = CNN_Net(), CNN_Net()

        self.e_greedy = EPSILON
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES_CNN * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        print(np.array(x).shape)
        x = np.reshape(x, (-1,N_CHANNEL*N_INPUT,INPUT_SIZE,INPUT_SIZE))
        # x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if use_cuda:
            x = Variable(torch.FloatTensor(x), 0).cuda()
        else:
            x = Variable(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.e_greedy:   # greedy
            actions_value = self.eval_net.forward(x)
            if use_cuda:
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            else:
                action = torch.max(actions_value, 1)[1].data.numpy()[0]     # return the argmax
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        s = s.flatten()
        s_ = s_.flatten()
        # print s.shape
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('Parameters updated')
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        if use_cuda:
            b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES_CNN]).cuda())
            b_a = Variable(torch.LongTensor(b_memory[:, N_STATES_CNN:N_STATES_CNN+1].astype(int)).cuda())
            b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES_CNN+1:N_STATES_CNN+2]).cuda())
            b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES_CNN:]).cuda())
        else:
            b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES_CNN]))
            b_a = Variable(torch.LongTensor(b_memory[:, N_STATES_CNN:N_STATES_CNN+1].astype(int)))
            b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES_CNN+1:N_STATES_CNN+2]))
            b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES_CNN:]))
        b_s = b_s.view(-1,N_CHANNEL * N_INPUT,INPUT_SIZE,INPUT_SIZE)
        b_s_ = b_s_.view(-1,N_CHANNEL * N_INPUT,INPUT_SIZE,INPUT_SIZE)
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.cpu().numpy()