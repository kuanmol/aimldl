import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from gymnasium.wrappers import RecordVideo
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# State Normalizer (from ppo_ant_v5_optimized5.py)
class StateNormalizer:
    def __init__(self, state_dim):
        self.mean = np.zeros(state_dim)
        self.var = np.ones(state_dim)
        self.count = 1e-4

    def update(self, state):
        self.count += 1
        delta = state - self.mean
        self.mean += delta / self.count
        delta2 = state - self.mean
        self.var += delta * delta2

    def normalize(self, state):
        return np.clip((state - self.mean) / np.sqrt(self.var / self.count + 1e-8), -10, 10)


# Actor-Critic Network (from ppo_ant_v5_optimized5.py)
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(action_std_init))

    def forward(self, state):
        value = self.critic(state)
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist, value

    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        dist, _ = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.cpu().numpy(), log_prob.item()


# PPO Agent (from ppo_ant_v5_optimized5.py)
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)


# Main video recording function
def main():
    # Create environment with video recording
    video_path = "D:/Artificial/AI/ppo/videos/"
    os.makedirs(video_path, exist_ok=True)
    env = gym.make("Ant-v5", render_mode="rgb_array")
    env = RecordVideo(env, video_path, video_length=1000, name_prefix="ant_v5_episode_2100")

    # Initialize agent and normalizer
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    normalizer = StateNormalizer(state_dim)
    agent = PPO(state_dim, action_dim)

    # Load checkpoint
    checkpoint_path = "D:/Artificial/AI/ppo/ppo_ant_episode_2100.pth"
    try:
        agent.policy.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} not found. Exiting.")
        return

    # Run one episode
    state, _ = env.reset()
    state = normalizer.normalize(state)
    episode_reward = 0
    done = False
    episode_timesteps = 0

    while not done and episode_timesteps < 1000:  # Max episode length
        action, _ = agent.policy.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        reward = np.clip(reward, -1, 1)  # Match training reward clipping
        next_state = normalizer.normalize(next_state)
        normalizer.update(next_state)
        done = terminated or truncated
        episode_reward += reward
        state = next_state
        episode_timesteps += 1

    print(f"Episode Reward: {episode_reward:.2f}, Timesteps: {episode_timesteps}")
    env.close()


if __name__ == "__main__":
    main()
