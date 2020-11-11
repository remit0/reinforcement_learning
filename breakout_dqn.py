from copy import deepcopy

import cv2 as cv
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class FrameTransformer:

    def crop(self, frame):
        return frame[31:195]

    def normalize(self, frame):
        return frame * (1 / 255)

    def to_gray(self, frame):
        return cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    def resize(self, frame):
        return cv.resize(frame, dsize=(80, 80), interpolation=cv.INTER_NEAREST)

    def transform(self, frame):
        frame = self.crop(frame)
        frame = self.to_gray(frame)
        frame = self.normalize(frame)
        frame = self.resize(frame)
        return frame


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(15 * 15 * 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 15 * 15 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Memory:

    def __init__(self, max_size):
        self.frames = np.empty((max_size, 80, 80), dtype=np.float32)
        self.actions = np.empty((max_size,), dtype=np.uint8)
        self.rewards = np.empty((max_size,), dtype=np.float32)
        self.dones = np.empty((max_size,), dtype=np.bool_)
        self.max_size = max_size
        self.n_insert = 0

    @property
    def insert_index(self):
        return self.n_insert % self.max_size

    @property
    def current_size(self):
        return min(self.n_insert, self.max_size)

    def get_state_indices(self, idx):
        return [(idx - i) % self.max_size for i in range(4)]

    def get_state(self, idx):
        state_indices = self.get_state_indices(idx)
        return self.frames[state_indices]

    def get_train_indices(self, batch_size):
        indices = []
        for i in range(batch_size):
            while True:
                index = np.random.randint(0, self.current_size)
                if index < 4:  # start of episode edge cases (need to see 4 frames first)
                    continue
                if self.dones[index - 4: index].any():  # there is a done flag in the middle of the series
                    continue
                break
            indices.append(index)
        return indices

    def get_batch(self, batch_size):
        indices = self.get_train_indices(batch_size)
        states = np.array([self.get_state(idx - 1) for idx in indices])
        next_states = np.array([self.get_state(idx) for idx in indices])
        return states, self.actions[indices], self.rewards[indices], self.dones[indices], next_states

    def insert(self, frame, action, reward, done):
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.dones[self.insert_index] = done
        self.frames[self.insert_index] = frame
        self.n_insert += 1


class DQN:

    def __init__(self, learning_rate):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.net = CNN().float().to(self.device)
        self.frozen_net = deepcopy(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def update_frozen_net(self):
        self.frozen_net = deepcopy(self.net)

    def predict(self, states, use_frozen):
        predictor = self.frozen_net if use_frozen else self.net
        states = torch.from_numpy(states).float().to(self.device)
        predictor.eval()
        with torch.no_grad():
            actions = predictor(states)
        return actions

    def one_hot_encode(self, actions):
        encoded = torch.zeros(len(actions), 4)
        for i, action in enumerate(actions):
            encoded[i][action] = 1
        return encoded.to(self.device)

    def cost(self, pred_action_values, actions, targets):
        action_values = torch.sum(pred_action_values * self.one_hot_encode(actions), dim=1)
        targets = torch.from_numpy(targets).float().to(self.device)
        cost = torch.sum(torch.square(targets - action_values))
        return cost

    def update(self, memory, batch_size, gamma):
        states, actions, rewards, dones, next_states = memory.get_batch(batch_size)
        next_q = np.max(self.predict(next_states, use_frozen=True).cpu().numpy(), axis=1)
        targets = rewards + gamma * next_q * np.invert(dones).astype(np.float32)

        self.net.train()
        self.optimizer.zero_grad()
        states = torch.from_numpy(states).float().to(self.device)
        pred_action_values = self.net(states)
        cost = self.cost(pred_action_values, actions, targets)
        cost.backward()
        self.optimizer.step()

    def sample_action(self, state, eps):
        if np.random.uniform(0, 1) < eps:
            return np.random.choice(4)
        else:
            state = np.expand_dims(state, axis=0)
            return np.argmax(self.predict(state, use_frozen=False).cpu()).item()


def update_state(state, next_frame):
    return np.append(state[1:, :, :], np.expand_dims(next_frame, 0), axis=0)


def compute_eps(n_episode, threshold):
    if n_episode > threshold:
        return 0.1
    else:
        return 1 - ((0.9 * n_episode) / threshold)


def init_memory(env, memory, transformer, size):
    while memory.current_size < size:
        env.reset()
        done = False
        while not done:
            action = np.random.choice(4)
            frame, reward, done, _ = env.step(action)
            frame = transformer.transform(frame)
            memory.insert(frame, action, reward, done)


def train_one_episode(env, dqn, transformer, memory, gamma, batch_size, eps, period, iters):
    frame = transformer.transform(env.reset())
    state = np.stack([frame] * 4, axis=0)
    done = False

    episode_reward = 0

    while not done:

        action = dqn.sample_action(state, eps)
        frame, reward, done, _ = env.step(action)
        frame = transformer.transform(frame)
        state = update_state(state, frame)
        memory.insert(frame, action, reward, done)
        dqn.update(memory, batch_size, gamma)

        if iters % period == 0:
            dqn.update_frozen_net()

        iters += 1
        episode_reward += reward

    return episode_reward, iters


def main():
    env = gym.envs.make("Breakout-v0").env

    gamma = 0.95
    period = 100
    learning_rate = 1e-5
    min_memory_size = 100
    max_memory_size = 10000
    batch_size = 64

    memory = Memory(max_memory_size)
    transformer = FrameTransformer()
    dqn = DQN(learning_rate)

    init_memory(env, memory, transformer, min_memory_size)

    num_episodes = 1000
    iters = 0
    rewards = []

    for n_ep in range(num_episodes):
        eps = compute_eps(n_ep, 5000)
        reward, iters = train_one_episode(env, dqn, transformer, memory, gamma, batch_size, eps, period, iters)
        rewards.append(reward)
        if n_ep % 100 == 0:
            print(f"avg reward at {n_ep}: {sum(rewards[-100:]) / 100}")


if __name__ == "__main__":
    main()
