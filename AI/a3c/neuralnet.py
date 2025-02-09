import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple


class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3, 3), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(512, 128)
        self.fc2a = torch.nn.Linear(128, action_size)
        self.fc2s = torch.nn.Linear(128, 1)

    def forward(self, state):
        x = self.conv1(state)
        F.relu(x)
        x = self.conv2(x)
        F.relu(x)
        x = self.conv3(x)
        F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        action_values = self.fc2a(x)
        state_value = self.fc2s(x)[0]
        return action_values,state_value

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

# env = gym.make('ALE/KungFuMaster-v5')
env = gym.make("KungFuMasterDeterministic-v0", render_mode="human")
env.reset()
print("Environment Loaded Successfully!")
env.close()
