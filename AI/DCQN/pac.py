import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision.transforms import Resize
from PIL import Image
from torchvision import transforms
import gymnasium as gym
import ale_py

# ✅ Register gym environments
gym.register_envs(ale_py)

# ✅ Create Gym Environment
env = gym.make('MsPacmanDeterministic-v0', full_action_space=False)
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ✅ Frame Preprocessing (Move Tensor to GPU)
def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return preprocess(frame).unsqueeze(0).to(device)

# ✅ Neural Network Model
class Network(nn.Module):
    def __init__(self, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(10 * 10 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ✅ Deep Q-Learning Agent
class Agent():
    def __init__(self, action_size):
        self.device = device
        self.action_size = action_size
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=5e-4)
        self.memory = deque(maxlen=10000)

    def step(self, state, action, reward, next_state, done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 100:
            experiences = random.sample(self.memory, k=100)
            self.learn(experiences, 0.99)

    def act(self, state, epsilon=0.):
        state = preprocess_frame(state).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experience, discount_factor):
        states, actions, rewards, next_states, dones = zip(*experience)
        states = torch.cat(states).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device).unsqueeze(1)
        rewards = torch.tensor(rewards).float().to(self.device).unsqueeze(1)
        next_states = torch.cat(next_states).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device).unsqueeze(1)
        next_qtarget = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (discount_factor * next_qtarget * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ✅ Initialize the Agent
agent = Agent(number_actions)

# ✅ Training Loop
number_episode = 2000
maximum_number_timesteps_per_episode = 10000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
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
    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_epsilons):.2f}', end="")
    if episode % 100 == 0:
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_epsilons):.2f}')
    if np.mean(scores_on_100_epsilons) >= 500.0:
        print(f'\nEnvironment solved in {episode - 100} episodes!\tAverage Score: {np.mean(scores_on_100_epsilons):.2f}')
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break

# ✅ Save a video of the trained model
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'MsPacmanDeterministic-v0')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data=f'''<video alt="test" autoplay loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{encoded.decode('ascii')}" type="video/mp4" />
             </video>'''))
    else:
        print("Could not find video")

show_video()
