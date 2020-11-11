import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):

    """ Feed forward network. """

    def __init__(self, state_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 3*state_dim)
        self.fc2 = nn.Linear(3*state_dim, state_dim)
        self.fc_mean = nn.Linear(state_dim, 1)
        self.fc_std = nn.Linear(state_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        std = F.softplus(std)
        return mean, std


class PolicyModel:

    """ Approximation of pi(a | s). """

    def __init__(self, state_dim):
        self.policy_net = NeuralNet(state_dim).double()

    def sample_action(self, state):
        state = torch.from_numpy(state).double()
        self.policy_net.eval()
        with torch.no_grad():
            mean, std = self.policy_net(state)
        return torch.clamp(torch.normal(mean, std), -1, 1)

    def perturb_params(self, noise_scale):
        for p in self.policy_net.parameters():
            p.requires_grad = False
            noise = torch.randn(*p.size())
            if np.random.uniform(0, 1) < 0.1:
                p = noise
            else:
                p += noise_scale * noise


def run_episode(env, policy_model, output_directory):
    if output_directory:
        env = gym.wrappers.Monitor(env, output_directory, force=True)

    state = env.reset()
    done = False
    episode_reward = 0
    iters = 0

    while not done and iters < 2000:
        # agent step
        action = policy_model.sample_action(state)
        state, reward, done, _ = env.step([action])

        # update variables
        episode_reward += reward
        iters += 1

    return episode_reward


def run_n_episodes(env, policy_model, n):
    total_reward = 0
    for _ in range(n):
        ep_reward = run_episode(env, policy_model, None)
        total_reward += ep_reward
    return total_reward / n


def main():

    # create environment
    env = gym.make("MountainCarContinuous-v0").env   # calling .env removes the 200 iterations limit

    # hyper parameters
    num_episodes = 100
    n_policy_evaluation = 3
    noise_scale = 0.5

    # loop variables
    best_reward = -float("inf")

    # instantiate policy model
    state_dim = env.observation_space.shape[0]
    policy_model = PolicyModel(state_dim)

    # hill climbing
    for n_ep in range(num_episodes):

        candidate = copy.deepcopy(policy_model)
        candidate.perturb_params(noise_scale)

        reward = run_n_episodes(env, candidate, n_policy_evaluation)

        if reward > best_reward:
            best_reward = reward
            policy_model = candidate
            print(f"New policy model found at episode {n_ep}, reward: {best_reward}")
        else:
            print(f"No improvement found at step {n_ep}, reward: {reward}")

    run_episode(env, policy_model, "output")


if __name__ == "__main__":
    main()
