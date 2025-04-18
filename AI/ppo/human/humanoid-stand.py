import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import pickle

# Updated Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 512
ENTROPY_COEF = 0.01
CONTROL_PENALTY = 0.01
GRAD_CLIP = 0.5
MIN_STD = 1e-6
MAX_EPISODE_STEPS = 1000
ENV_NAME = "HumanoidStandup-v4"
TOTAL_TIMESTEPS = 10000000
REWARD_THRESHOLD = 45000
THRESHOLD_EPISODES = 3
PLATEAU_WINDOW = 50
PLATEAU_IMPROVEMENT = 0.001

class RunningStat:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)
        self.count = 0

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_std = np.std(x, axis=0) + MIN_STD
        batch_count = x.shape[0]
        self.count += batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / self.count
        m_a = self.std * self.std * (self.count - batch_count)
        m_b = batch_std * batch_std * batch_count
        self.std = np.sqrt((m_a + m_b + np.square(delta) * self.count * batch_count / self.count) / self.count)

    def normalize(self, x):
        normalized = (x - self.mean) / self.std
        return np.clip(normalized, -5, 5)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, act_dim * 2)
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        x = self.net(state)
        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -10, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        if torch.isnan(mean).any() or torch.isnan(log_std).any():
            return None, None, None
        std = log_std.exp()
        dist = Normal(mean, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def get_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        pre_tanh = dist.sample()
        return torch.tanh(pre_tanh)

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net(state)

class PPO:
    def __init__(self, obs_dim, act_dim, device):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.device = device

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(EPOCHS):
            for idx in range(0, len(states), BATCH_SIZE):
                batch_idx = slice(idx, idx + BATCH_SIZE)
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                action_out, log_prob, entropy = self.actor.sample(batch_states)
                if action_out is None:
                    continue
                ratio = torch.exp(log_prob - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy.mean()

                value = self.critic(batch_states).squeeze()
                critic_loss = (batch_returns - value).pow(2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
                self.critic_optimizer.step()

    def save(self, path, obs_normalizer):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
        with open(path + '.norm', 'wb') as f:
            pickle.dump(obs_normalizer, f)

def train():
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPO(obs_dim, act_dim, device)
    obs_normalizer = RunningStat(obs_dim)

    total_steps = 0
    episode_rewards = []
    threshold_count = 0
    plateau_rewards = []

    while total_steps < TOTAL_TIMESTEPS:
        state, _ = env.reset()
        episode_reward = 0
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        for t in range(MAX_EPISODE_STEPS):
            obs_normalizer.update(np.array([state]))
            norm_state = obs_normalizer.normalize(state)
            state_tensor = torch.FloatTensor(norm_state).to(device)

            action, log_prob, _ = agent.actor.sample(state_tensor.unsqueeze(0))
            if action is None:
                break
            value = agent.critic(state_tensor.unsqueeze(0)).item()

            action_np = action.detach().cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            reward -= CONTROL_PENALTY * np.sum(action_np ** 2)
            done = terminated or truncated

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value)
            dones.append(done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                break

        if np.isnan(next_state).any():
            continue

        obs_normalizer.update(np.array([next_state]))
        norm_next_state = obs_normalizer.normalize(next_state)
        next_value = agent.critic(torch.FloatTensor(norm_next_state).to(device).unsqueeze(0)).item()

        advantages, returns = agent.compute_gae(rewards, values, next_value, dones)
        agent.update(states, actions, log_probs, returns, advantages)

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        print(f"Episode: {len(episode_rewards)}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")

        if len(episode_rewards) >= 10 and avg_reward >= REWARD_THRESHOLD:
            threshold_count += 1
            if threshold_count >= THRESHOLD_EPISODES:
                print(f"Stopping training: Avg reward {avg_reward:.2f} >= {REWARD_THRESHOLD}")
                agent.save(f"ppo_humanoidstandup_final.pth", obs_normalizer)
                break
        else:
            threshold_count = 0

        if len(episode_rewards) >= PLATEAU_WINDOW + 10:
            old_avg = np.mean(plateau_rewards[-PLATEAU_WINDOW - 10:-PLATEAU_WINDOW])
            new_avg = np.mean(plateau_rewards[-PLATEAU_WINDOW:])
            improvement = (new_avg - old_avg) / old_avg if old_avg != 0 else 0
            if improvement < PLATEAU_IMPROVEMENT:
                print(f"Stopping training: Reward plateaued (improvement {improvement:.4f} < {PLATEAU_IMPROVEMENT})")
                agent.save(f"ppo_humanoidstandup_final.pth", obs_normalizer)
                break
            plateau_rewards.append(avg_reward)

        if len(episode_rewards) % 10 == 0:
            agent.save(f"ppo_humanoidstandup_{len(episode_rewards)}.pth", obs_normalizer)
    env.close()

if __name__ == "__main__":
    train()