import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class NeuralNet(nn.Module):

    """ Feed forward network. """

    def __init__(self, layer_list, use_softmax):
        super(NeuralNet, self).__init__()
        self.use_softmax = use_softmax
        self.layers = nn.ModuleList()
        for i in range(len(layer_list)-1):
            self.layers.append(nn.Linear(layer_list[i], layer_list[i+1]))

    def forward(self, x):
        for layer in self.layers:
            # x = F.relu(layer(x))
            x = layer(x)
        if self.use_softmax:
            x = F.softmax(x, dim=0)
        return x


class PolicyModel:

    """ Approximation of pi(a | s). """

    def __init__(self, layer_list, lr):
        self.layer_list = layer_list
        self.policy_net = NeuralNet(layer_list, use_softmax=True).double()
        self.optimizer = optim.Adagrad(self.policy_net.parameters(), lr=lr)

    def cost(self, p_a_given_s, action, advantage):
        log_pi = torch.log(p_a_given_s[action])
        cost = -1 * advantage * log_pi
        return cost

    def partial_fit(self, state, action, advantage):
        state = torch.from_numpy(state).double()
        self.policy_net.train()
        self.optimizer.zero_grad()
        p_a_given_s = self.policy_net(state)
        cost = self.cost(p_a_given_s, action, advantage)
        cost.backward()
        self.optimizer.step()

    def predict(self, state):
        state = torch.from_numpy(state).double()
        self.policy_net.eval()
        with torch.no_grad():
            action_probs = self.policy_net(state)
        return action_probs

    def sample_action(self, state):
        action_probs = self.predict(state)
        return np.random.choice(len(action_probs), p=action_probs)


class ValueModel:

    """ Approximation of V(s). """

    def __init__(self, layer_list, lr):
        self.value_net = NeuralNet(layer_list, use_softmax=False).double()
        self.optimizer = optim.SGD(self.value_net.parameters(), lr=lr)

    def cost(self, pred_value, target_value):
        return torch.square(target_value - pred_value)

    def partial_fit(self, state, target_value):
        state = torch.from_numpy(state).double()
        self.value_net.train()
        self.optimizer.zero_grad()
        pred_value = self.value_net(state)
        loss = self.cost(pred_value, target_value)
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        state = torch.from_numpy(state).double()
        self.value_net.eval()
        with torch.no_grad():
            value = self.value_net(state)
        return value


def run_training_episode(env, policy_model, value_model, gamma):
    """ TD(0) Policy Gradient episode (does not learn with current parameters :( ). """
    state = env.reset()
    done = False
    episode_reward = 0
    iters = 0

    while not done and iters < 2000:
        # agent step
        action = policy_model.sample_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward -= 200

        # advantage
        v_s = value_model.predict(state)
        v_s_prime = value_model.predict(next_state)
        G = reward + gamma * v_s_prime
        advantage = G - v_s

        # update models
        policy_model.partial_fit(state, action, advantage)
        value_model.partial_fit(state, G)

        # update variables
        episode_reward += reward
        state = next_state
        iters += 1

    return episode_reward


def run_training_episode_mc(env, policy_model, value_model, gamma):
    """ MonteCarlo Policy Gradient episode. """
    state = env.reset()
    done = False
    episode_reward = 0
    iters = 0

    states = []
    actions = []
    rewards = []
    reward = 0

    while not done and iters < 2000:

        action = policy_model.sample_action(state)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state, reward, done, _ = env.step(action)
        if done:
            reward = -200

        episode_reward += reward
        iters += 1

    returns = []
    advantages = []
    G = 0

    for s, r in zip(reversed(states), reversed(rewards)):

        returns.append(G)
        advantages.append(G - value_model.predict(s))
        G = r + gamma * G

    returns.reverse()
    advantages.reverse()

    for s, a, ad, r in zip(states, actions, advantages, returns):
        policy_model.partial_fit(s, a, ad)
        value_model.partial_fit(s, r)

    return episode_reward


def run_episode(env, policy_model, output_directory):
    """ test a trained agent. """
    env = gym.wrappers.Monitor(env, output_directory, force=True)
    state = env.reset()
    done = False
    iters = 0

    while not done and iters < 2000:
        action = policy_model.sample_action(state)
        state, _, done, _ = env.step(action)
        iters += 1
    print(f"Ended after {iters} iterations.")


def main():

    # create environment
    env = gym.make("CartPole-v0").env   # calling .env removes the 200 iterations limit

    # hyper parameters
    num_episodes = 1000
    gamma = 0.95
    policy_lr = 1e-1
    value_lr = 1e-4
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # instantiate policy model
    layers = [state_dim, 3 * state_dim, state_dim, action_dim]
    policy_model = PolicyModel(layers, policy_lr)

    # instantiate value model
    layers = [state_dim, 3 * state_dim, state_dim, 1]
    value_model = ValueModel(layers, value_lr)

    # training
    rewards = []
    for n_ep in range(num_episodes):
        ep_reward = run_training_episode_mc(env, policy_model, value_model, gamma)
        rewards.append(ep_reward)
        if n_ep % 100 == 0:
            print(f"avg reward at episode {n_ep}: {np.mean(rewards[-100:])}")

    # testing
    run_episode(env, policy_model, output_directory="output")


if __name__ == "__main__":
    main()
