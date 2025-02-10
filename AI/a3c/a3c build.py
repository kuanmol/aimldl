import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
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
        return action_values, state_value


import gymnasium as gym
import ale_py
import cv2
import math
import random
import numpy as np

gym.register_envs(ale_py)


class PreprocessAtari(ObservationWrapper):

    def __init__(self, env, height=42, width=42, crop=lambda img: img, dim_order='pytorch', color=False, n_frames=4):
        super(PreprocessAtari, self).__init__(env)
        self.img_size = (height, width)
        self.crop = crop
        self.dim_order = dim_order
        self.color = color
        self.frame_stack = n_frames
        n_channels = 3 * n_frames if color else n_frames
        obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.frames = np.zeros(obs_shape, dtype=np.float32)

    def reset(self):
        self.frames = np.zeros_like(self.frames)
        obs, info = self.env.reset()
        self.update_buffer(obs)
        return self.frames, info

    def observation(self, img):
        img = self.crop(img)
        img = cv2.resize(img, self.img_size)
        if not self.color:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32') / 255.
        if self.color:
            self.frames = np.roll(self.frames, shift=-3, axis=0)
        else:
            self.frames = np.roll(self.frames, shift=-1, axis=0)
        if self.color:
            self.frames[-3:] = img
        else:
            self.frames[-1] = img
        return self.frames

    def update_buffer(self, obs):
        self.frames = self.observation(obs)


def make_env():
    env = gym.make("KungFuMasterDeterministic-v0", render_mode="rgb_array")

    env = PreprocessAtari(env, height=42, width=42, crop=lambda img: img, dim_order='pytorch', color=False, n_frames=4)
    return env


env = make_env()

state_shape = env.observation_space.shape
number_actions = env.action_space.n
print("State shape:", state_shape)
print("Number actions:", number_actions)
print("Action names:", env.unwrapped.get_action_meanings())

# intializing the hyperparameters
learning_rate = 1e-4
discount_factor = 0.99
number_environment = 10


class Agent():
    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Network(action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def act(self, state):
        if state.ndim == 3:
            state = [state]
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_values, _ = self.network(state)
        policy = F.softmax(action_values, dim=-1)
        return np.array([np.random.choice(len(p), p=p) for p in policy.detach().cpu().numpy()])



