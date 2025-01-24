import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import imageio
import glob
import io
import base64
from IPython.display import HTML, display
import gymnasium as gym

class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


# Replay Memory
class ReplayMemory(object):
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        experience = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experience if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experience if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experience if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experience if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experience if e is not None]).astype(np.uint8)).float().to(
            self.device)
        return states, next_states, actions, rewards, dones


# Agent Class
class Agent():
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=5e-4)
        self.memory = ReplayMemory(int(1e5))
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > 100:
                experiences = self.memory.sample(100)
                self.learn(experiences, 0.99)

    def act(self, state, epsilon=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experience, discount_factor):
        states, next_state, actions, rewards, dones = experience
        next_qtarget = self.target_qnetwork(next_state).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (discount_factor * next_qtarget * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, 1e-3)

    def soft_update(self, local_model, target_model, interpolation_parameter):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)


# Function to Record Video of the Agent
def show_video_of_model(agent, env_name, episode, save_path="videos"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    video_filename = os.path.join(save_path, f"episode_{episode}.mp4")
    imageio.mimsave(video_filename, frames, fps=30)
    env.close()
    print(f"Video saved for episode {episode} at {video_filename}")


# Function to Display Videos
def show_video():
    mp4list = glob.glob('videos/*.mp4')
    if len(mp4list) > 0:
        for video_path in mp4list:
            video = io.open(video_path, 'r+b').read()
            encoded = base64.b64encode(video)
            display(HTML(data=f'''
                <video alt="episode video" autoplay loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{encoded.decode('ascii')}" type="video/mp4" />
                </video>
                '''))
    else:
        print("Could not find videos")


# Training Loop
env = gym.make("LunarLander-v3")
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n

agent = Agent(state_size, number_actions)

number_episode = 2000
maximum_number_timesteps_per_episode = 1000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995
epsilon = epsilon_starting_value
scores_on_100_epsilons = deque(maxlen=100)

for episode in range(1, number_episode + 1):
    state, _ = env.reset()
    score = 0
    for t in range(maximum_number_timesteps_per_episode):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break

    scores_on_100_epsilons.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)

    if episode % 100 == 0:
        print(f'\nEpisode {episode}\tAverage Score: {np.mean(scores_on_100_epsilons):.2f}')
        show_video_of_model(agent, 'LunarLander-v3', episode)

    if np.mean(scores_on_100_epsilons) >= 200.0:
        print(f'\nEnvironment solved in {episode - 100} episodes! Average Score: {np.mean(scores_on_100_epsilons):.2f}')
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break

show_video()
