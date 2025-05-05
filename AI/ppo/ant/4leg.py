import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPSILON = 0.2
CRITIC_LOSS_WEIGHT = 0.5
ENTROPY_BETA = 0.1
PPO_EPOCHS = 8
BATCH_SIZE = 256
MAX_TIMESTIPS = 3000000
LEARNING_RATE = 3e-4
REWARD_THRESHOLD = 3000.0


# State Normalizer
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


# Actor-Critic Network
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

    def evaluate(self, states, actions):
        dist, values = self.forward(states)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, values.squeeze(-1), entropy


# PPO Agent
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.mse_loss = nn.MSELoss()

    def update(self, memory):
        states, actions, log_probs_old, returns, advantages = memory
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_policy_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_batches = 0

        for _ in range(PPO_EPOCHS):
            for i in range(0, len(states), BATCH_SIZE):
                batch_idx = slice(i, i + BATCH_SIZE)
                state_batch = torch.FloatTensor(states[batch_idx]).to(device)
                action_batch = torch.FloatTensor(actions[batch_idx]).to(device)
                log_probs_old_batch = torch.FloatTensor(log_probs_old[batch_idx]).to(device)
                return_batch = torch.FloatTensor(returns[batch_idx]).to(device)
                adv_batch = torch.FloatTensor(advantages[batch_idx]).to(device)

                log_probs, values, entropy = self.policy.evaluate(state_batch, action_batch)
                ratios = torch.exp(log_probs - log_probs_old_batch)
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(values, return_batch)
                entropy_loss = ENTROPY_BETA * entropy.mean()
                loss = policy_loss + CRITIC_LOSS_WEIGHT * critic_loss - entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        return (total_policy_loss / num_batches, total_critic_loss / num_batches, total_entropy / num_batches)


# Main training loop
def main():
    env = gym.make("Ant-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    normalizer = StateNormalizer(state_dim)
    agent = PPO(state_dim, action_dim)
    total_timesteps = 0
    episode = 0
    reward_window = []
    WINDOW_SIZE = 50
    reward_threshold_reached = False

    # Try loading checkpoint
    checkpoints = ["ppo_ant_episode_2100.pth", "ppo_ant_episode_2000.pth", "ppo_ant_episode_1900.pth"]
    for checkpoint in checkpoints:
        try:
            agent.policy.load_state_dict(torch.load(checkpoint))
            print(f"Loaded model from {checkpoint}")
            episode = int(checkpoint.split("_")[-1].split(".")[0])
            break
        except FileNotFoundError:
            print(f"No {checkpoint} found, trying next or starting fresh")

    while total_timesteps < MAX_TIMESTIPS:
        state, _ = env.reset()
        state = normalizer.normalize(state)
        episode_reward = 0
        done = False
        states, actions, log_probs, rewards = [], [], [], []
        episode_timesteps = 0

        while not done and total_timesteps < MAX_TIMESTIPS:
            action, log_prob = agent.policy.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward = np.clip(reward, -1, 1)  # Clip rewards
            next_state = normalizer.normalize(next_state)
            normalizer.update(next_state)
            done = terminated or truncated
            episode_reward += reward
            total_timesteps += 1
            episode_timesteps += 1

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

            if episode_timesteps % BATCH_SIZE == 0 or done:
                with torch.no_grad():
                    _, next_value = agent.policy.forward(torch.FloatTensor(state).to(device))
                    next_value = next_value.item() if not done else 0
                returns = []
                advantages = []
                gae = 0
                for r in reversed(rewards):
                    delta = r + GAMMA * next_value - next_value
                    gae = delta + GAMMA * LAMBDA * (1 - done) * gae
                    advantages.insert(0, gae)
                    returns.insert(0, gae + next_value)
                    next_value = next_value

                states_tensor = np.array(states)
                actions_tensor = np.array(actions)
                advantages = np.array(advantages)
                returns = np.array(returns)
                policy_loss, critic_loss, entropy = agent.update(
                    (states_tensor, actions, log_probs, returns, advantages))
                states, actions, log_probs, rewards = [], [], [], []

        episode += 1
        print(
            f"Episode {episode}, Reward: {episode_reward:.2f}, Timesteps: {total_timesteps}, Length: {episode_timesteps}")
        print(f"Avg Policy Loss: {policy_loss:.4f}, Avg Value Loss: {critic_loss:.4f}, Avg Entropy: {entropy:.4f}")
        reward_window.append(episode_reward)
        if len(reward_window) > WINDOW_SIZE:
            reward_window.pop(0)
            avg_reward = np.mean(reward_window)
            print(f"Average Reward over {WINDOW_SIZE} episodes: {avg_reward:.2f}")
            if avg_reward >= REWARD_THRESHOLD and not reward_threshold_reached:
                print(f"Saving model at Episode {episode} with average reward {avg_reward:.2f}")
                torch.save(agent.policy.state_dict(), f"ppo_ant_reward_3000_episode_{episode}.pth")
                reward_threshold_reached = True
                print(f"Stopping training: Average reward {avg_reward:.2f} >= {REWARD_THRESHOLD}")
                break
        if episode % 100 == 0:
            torch.save(agent.policy.state_dict(), f"ppo_ant_episode_{episode}.pth")

    env.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()
