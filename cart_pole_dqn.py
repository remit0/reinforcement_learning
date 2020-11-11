import gym
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from copy import deepcopy


class NeuralNet(nn.Module):

    """ Feed forward network. """

    def __init__(self, layers):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DQN:

    def __init__(self, layers, learning_rate, exp_size, btc_size):
        self.layers = layers
        self.net = NeuralNet(layers).double()
        self.memory_net = deepcopy(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.exp_size = exp_size
        self.btc_size = btc_size
        self.experiences = {"s": [], "a": [], "r": [], "s_prime": [], "done": []}

    def add_experience(self, state, action, reward, next_state, done):
        if len(self.experiences["s"]) >= self.exp_size[1]:
            self.experiences["s"].pop(0)
            self.experiences["a"].pop(0)
            self.experiences["r"].pop(0)
            self.experiences["s_prime"].pop(0)
            self.experiences["done"].pop(0)
        self.experiences["s"].append(state)
        self.experiences["a"].append(action)
        self.experiences["r"].append(reward)
        self.experiences["s_prime"].append(next_state)
        self.experiences["done"].append(done)

    def one_hot_encode(self, actions):
        encoded = torch.zeros(len(actions), self.layers[-1])
        for i, action in enumerate(actions):
            encoded[i][action] = 1
        return encoded

    def cost(self, pred_action_values, actions, targets):
        action_values = torch.sum(pred_action_values * self.one_hot_encode(actions), dim=1)
        targets = torch.from_numpy(np.array(targets)).double()
        cost = torch.sum(torch.square(targets - action_values))
        return cost

    def update(self, gamma):

        if len(self.experiences["s"]) < self.exp_size[0]:
            return
        # todo replace by a torch.dataset.Dataset
        indices = np.random.choice(len(self.experiences["s"]), size=self.btc_size, replace=False)
        states = [self.experiences["s"][i] for i in indices]
        actions = [self.experiences["a"][i] for i in indices]
        rewards = [self.experiences["r"][i] for i in indices]
        next_states = [self.experiences["s_prime"][i] for i in indices]
        dones = [self.experiences["done"][i] for i in indices]
        next_Q = np.max(self.predict(next_states, use_memory=True).numpy(), axis=1)
        targets = [r + gamma * q if not done else r for r, q, done in zip(rewards, next_Q, dones)]

        self.net.train()
        self.optimizer.zero_grad()
        states = torch.from_numpy(np.array(states)).double()
        pred_action_values = self.net(states)
        cost = self.cost(pred_action_values, actions, targets)
        cost.backward()
        self.optimizer.step()

    def predict(self, state, use_memory):
        state = torch.from_numpy(np.array(state)).double()
        predictor = self.memory_net if use_memory else self.net
        predictor.eval()
        with torch.no_grad():
            actions = predictor(state)
        return actions

    def update_memory(self):
        self.memory_net = deepcopy(self.net)

    def sample_action(self, state, eps):
        if np.random.uniform(0, 1) < eps:
            return np.random.randint(0, 2)
        else:
            return np.argmax(self.predict(state, use_memory=False)).item()


def run_one_episode(env, dqn, gamma, eps, update_period):
    state = env.reset()
    done = False
    episode_reward = 0
    iters = 0

    while not done and iters < 2000:
        # run one step
        action = dqn.sample_action(state, eps)
        next_state, reward, done, _ = env.step(action)

        # update Q model
        dqn.add_experience(state, action, reward, next_state, done)
        dqn.update(gamma)

        # update target model
        if iters % update_period == 0:
            dqn.update_memory()

        # update variables
        state = next_state
        iters += 1
        episode_reward += reward

    return episode_reward


def main():

    # environment
    env = gym.make("CartPole-v0").env
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]

    # hyper parameters
    gamma = 0.99
    eps = 0.1
    update_period = 50
    dqn_lr = 1e-2
    q_layers = [state_dim, 20, 10, action_dim]
    exp_size = (100, 10000)
    batch_size = 32

    # variables
    dqn = DQN(q_layers, dqn_lr, exp_size, batch_size)
    num_episodes = 3000
    rewards = []

    # main loop
    for n_episode in range(num_episodes):
        reward = run_one_episode(env, dqn, gamma, eps, update_period)
        rewards.append(reward)
        if n_episode % 100 == 0:
            print(f"avg reward at {n_episode}: {sum(rewards[-100:]) / 100}")


if __name__ == "__main__":
    main()
