import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import threading
import multiprocessing
import os
from wrappers import wrap_atari

# Hyperparameters
NUM_WORKERS = 10
GAMMA = 0.99
T_MAX = 5
LEARNING_RATE = 7e-4
MAX_EPISODES = 10000
TARGET_SCORE = 1000
ENTROPY_BETA = 0.01
CLIP_GRAD = 40

# Environment settings
ENV_NAME = "KungFuMasterNoFrameskip-v4"
FRAME_STACK = 4

# Global variables
global_episode = multiprocessing.Value('i', 0)
global_scores = deque(maxlen=100)
best_score = -float('inf')


def record_video(env_id, model, video_folder='videos'):
    """
    Record a video using the trained model
    """
    env = wrap_atari(gym.make(env_id))
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)

    model.eval()
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(state_tensor)
        prob = F.softmax(logits, dim=-1)
        action = torch.argmax(prob).item()

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    env.close()
    print(f"Recorded episode with reward: {total_reward}")
    return total_reward

class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(FRAME_STACK, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc_pi = nn.Linear(512, num_actions)
        self.fc_v = nn.Linear(512, 1)

        self.num_actions = num_actions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        pi = self.fc_pi(x)
        v = self.fc_v(x)

        return pi, v


def worker(worker_id, global_model, optimizer, global_episode, global_scores, best_score):
    torch.manual_seed(worker_id)
    env = wrap_atari(gym.make(ENV_NAME))
    local_model = ActorCritic(env.action_space.n)
    local_model.load_state_dict(global_model.state_dict())

    t_max = T_MAX
    t = 0
    scores = []

    while global_episode.value < MAX_EPISODES:
        state = env.reset()
        done = False
        episode_reward = 0
        hx = torch.zeros(1, 512)

        while not done:
            t += 1
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
            logits, value = local_model(state_tensor)
            prob = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)

            m = Categorical(prob)
            action = m.sample().item()

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            if t >= t_max or done:
                with global_episode.get_lock():
                    global_episode.value += 1

                # Update global model
                R = torch.zeros(1, 1) if done else local_model(
                    torch.FloatTensor(np.array(next_state)).unsqueeze(0))[1].detach()

                # Compute losses and update
                optimizer.zero_grad()

                # Compute policy and value losses
                advantage = R - value
                policy_loss = -(log_prob[0, action] * advantage.detach())
                value_loss = F.smooth_l1_loss(value, R.detach())
                loss = policy_loss + 0.5 * value_loss - ENTROPY_BETA * entropy

                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), CLIP_GRAD)

                # Transfer gradients to global model
                for local_param, global_param in zip(local_model.parameters(),
                                                     global_model.parameters()):
                    if global_param.grad is not None:
                        break
                    global_param._grad = local_param.grad

                optimizer.step()
                local_model.load_state_dict(global_model.state_dict())
                t = 0

                if done:
                    scores.append(episode_reward)
                    with global_episode.get_lock():
                        if len(scores) > 0:
                            avg_score = np.mean(scores[-100:])
                            if avg_score > best_score.value:
                                best_score.value = avg_score
                                torch.save(global_model.state_dict(),
                                           f'kungfu_a3c_best_{avg_score:.2f}.pth')

                            print(f"Worker {worker_id}, Episode {global_episode.value}, "
                                  f"Score: {episode_reward}, Avg: {avg_score:.2f}, "
                                  f"Best: {best_score.value:.2f}")

                            if avg_score >= TARGET_SCORE:
                                print(f"Solved! Average score: {avg_score:.2f}")
                                return
                    break

            state = next_state


def main():
    env = wrap_atari(gym.make(ENV_NAME))
    global_model = ActorCritic(env.action_space.n)
    global_model.share_memory()

    optimizer = optim.Adam(global_model.parameters(), lr=LEARNING_RATE)

    workers = []
    for worker_id in range(NUM_WORKERS):
        worker_args = (worker_id, global_model, optimizer,
                       global_episode, global_scores, best_score)
        w = multiprocessing.Process(target=worker, args=worker_args)
        w.start()
        workers.append(w)

    for w in workers:
        w.join()


if __name__ == "__main__":
    main()