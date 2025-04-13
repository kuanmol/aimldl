import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gymnasium as gym
import ale_py
import cv2
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import tqdm
import os
import imageio
from IPython.display import HTML, display
import base64
import io
import glob

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Register Atari environments
gym.register_envs(ale_py)


class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 128)
        self.fc2a = nn.Linear(128, action_size)
        self.fc2s = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        action_scores = self.fc2a(x)
        state_value = self.fc2s(x)
        return F.softmax(action_scores, dim=-1), state_value


class PreprocessAtari(ObservationWrapper):
    def __init__(self, env, height=42, width=42, dim_order='pytorch', color=False, n_frames=4):
        super(PreprocessAtari, self).__init__(env)
        self.img_size = (height, width)
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
        return self.observation(obs), info

    def observation(self, img):
        img = img[34:-16, :, :]  # Crop
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


def make_env():
    env = gym.make("KungFuMaster-v4", render_mode="rgb_array", frameskip=1)
    env = PreprocessAtari(env, height=42, width=42, dim_order='pytorch', color=False, n_frames=4)
    return env


# Hyperparameters
learning_rate = 1e-4
discount_factor = 0.99
n_workers = 4
t_max = 5
max_episodes = 10000
target_score = 3000  # Stop when this score is reached


class Worker(mp.Process):
    def __init__(self, global_network, optimizer, global_ep_idx, res_queue, worker_id, action_size):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.global_ep_idx = global_ep_idx
        self.res_queue = res_queue
        self.global_network = global_network
        self.optimizer = optimizer
        self.local_network = Network(action_size)
        self.action_size = action_size

    def run(self):
        env = make_env()
        total_step = 1

        while self.global_ep_idx.value < max_episodes:
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                self.local_network.load_state_dict(self.global_network.state_dict())

                states = []
                actions = []
                rewards = []
                values = []

                for _ in range(t_max):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs, value = self.local_network(state_tensor)

                    action = torch.multinomial(action_probs, 1).item()
                    next_state, reward, done, _, _ = env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    values.append(value.item())

                    state = next_state
                    episode_reward += reward
                    total_step += 1

                    if done:
                        break

                R = 0 if done else self.local_network(torch.FloatTensor(state).unsqueeze(0))[1].item()
                returns = []
                for r in rewards[::-1]:
                    R = r + discount_factor * R
                    returns.insert(0, R)

                returns = torch.FloatTensor(returns)
                values = torch.FloatTensor(values)
                advantages = returns - values

                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions)

                self.optimizer.zero_grad()
                action_probs, value_preds = self.local_network(states)

                log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
                policy_loss = -(log_probs * advantages.detach()).mean()
                value_loss = F.mse_loss(returns, value_preds.squeeze(1))
                entropy = -(action_probs * torch.log(action_probs)).sum(1).mean()
                total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                total_loss.backward()

                for local_param, global_param in zip(self.local_network.parameters(),
                                                     self.global_network.parameters()):
                    if global_param.grad is not None:
                        global_param._grad = local_param.grad

                self.optimizer.step()

            with self.global_ep_idx.get_lock():
                self.global_ep_idx.value += 1

            self.res_queue.put(episode_reward)
            print(f"Worker {self.worker_id} finished episode {self.global_ep_idx.value} with reward {episode_reward}")

            # Early stopping condition
            if episode_reward >= target_score:
                print(f"\n🎯 Target score {target_score} reached! Stopping training.")
                break

        env.close()


class A3CAgent:
    def __init__(self):
        self.env = make_env()
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.global_network = Network(self.action_size)
        self.global_network.share_memory()
        self.optimizer = torch.optim.Adam(self.global_network.parameters(), lr=learning_rate)

    def train(self):
        global_ep_idx = mp.Value('i', 0)
        res_queue = mp.Queue()

        workers = [Worker(self.global_network, self.optimizer, global_ep_idx, res_queue, i, self.action_size)
                   for i in range(n_workers)]

        [w.start() for w in workers]

        scores = []
        try:
            while global_ep_idx.value < max_episodes:
                reward = res_queue.get()
                if reward is not None:
                    scores.append(reward)
                    if reward >= target_score:
                        break
        finally:
            [w.terminate() for w in workers]
            return scores

    def evaluate(self, n_episodes=3, render=True):
        env = make_env()
        rewards = []
        frames = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                if render:
                    frames.append(env.render())

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, _ = self.global_network(state_tensor)
                action = torch.argmax(action_probs).item()

                state, reward, done, _, _ = env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            print(f"Evaluation episode reward: {episode_reward}")

        env.close()

        if frames:
            self._save_video(frames)

        return np.mean(rewards)

    def _save_video(self, frames, fps=30):
        os.makedirs("videos", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"videos/kungfu_{timestamp}.mp4"
        imageio.mimsave(filename, frames, fps=fps)
        print(f"Saved video to {filename}")

        # Display video in notebook
        self._display_video(filename)

    def _display_video(self, filename):
        mp4list = glob.glob(filename)
        if len(mp4list) > 0:
            mp4 = mp4list[0]
            video = io.open(mp4, 'r+b').read()
            encoded = base64.b64encode(video)
            display(HTML(data='''<video alt="test" autoplay loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'))))
        else:
            print("Could not find video")


if __name__ == "__main__":
    from datetime import datetime

    agent = A3CAgent()
    print("Starting training...")
    scores = agent.train()

    print("\nTraining complete. Evaluating...")
    mean_reward = agent.evaluate(n_episodes=3, render=True, save_video=True)
    print(f"Mean evaluation reward: {mean_reward}")