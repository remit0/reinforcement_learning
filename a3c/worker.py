from collections import namedtuple

import numpy as np
import torch

from a3c.imgproc import FrameTransformer
from a3c.nets import create_networks

Step = namedtuple("Step", ["state", "action", "reward", "next_state", "done"])


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def repeat_frame(frame):
    return np.stack([frame] * 4, axis=0)


def shift_frames(state, next_frame):
    return np.append(state[1:, :, :], np.expand_dims(next_frame, 0), axis=0)


def one_hot_encode(actions):
    encoded = torch.zeros(len(actions), 2)
    for i, action in enumerate(actions):
        encoded[i][action] = 1
    return encoded


def policy_cost(pred_actions, actions, advantages, c):
    entropy = - torch.sum(pred_actions * torch.log(pred_actions), dim=1)

    actions = one_hot_encode(actions)
    selected_actions = torch.sum(pred_actions * actions, dim=1)
    cost = advantages * torch.log(selected_actions) + c * entropy
    cost = torch.sum(cost)
    return cost


def value_cost(pred_values, target_values):
    return torch.sum(torch.square(pred_values - target_values))


class Worker:

    def __init__(self, name, env, gamma):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.transformer = FrameTransformer()
        self.policy, self.value = create_networks()
        # init state
        self.env.reset()
        self.state = repeat_frame(self.transformer.transform(self.env.render(mode='rgb_array')))

    def predict(self, states, type_):
        net = self.policy if type_ == "action" else self.value
        states = torch.from_numpy(states).float()
        net.eval()
        with torch.no_grad():
            evaluations = net(states)
        return evaluations

    def get_action(self, state):
        action_probas = self.predict(np.expand_dims(state, axis=0), "action")
        action_probas = action_probas[0].numpy()
        return np.random.choice(len(action_probas), p=action_probas)

    def get_value(self, state):
        value = self.predict(state.expand_dims(axis=0), "value")
        return value.item()

    def run_n_steps(self, n, lock, counter):
        steps = []
        for _ in range(n):
            action = self.get_action(self.state)
            _, reward, done, _ = self.env.step(action)
            next_state = shift_frames(self.state, self.transformer.transform(self.env.render(mode='rgb_array')))
            step = Step(self.state, action, reward, next_state, done)
            steps.append(step)

            with lock:
                counter.value += 1

            if done:
                self.env.reset()
                self.state = repeat_frame(self.transformer.transform(self.env.render(mode='rgb_array')))
                break
            else:
                self.state = next_state

        return steps

    def get_batch(self, steps):
        reward = 0
        if not steps[-1].done:
            reward = self.get_value(steps[-1].next_state)

        states = []
        advantages = []
        targets = []
        actions = []
        for step in reversed(steps):
            reward = step.reward + self.gamma * reward
            advantage = reward - self.get_value(step.state)
            states.append(step.state)
            actions.append(step.action)
            advantages.append(advantage)
            targets.append(reward)
        return states, actions, advantages, targets

    def update(self, steps, shared_policy, shared_value, p_lr, v_lr):
        states, actions, advantages, targets = self.get_batch(steps)

        # optimizers
        p_optim = torch.optim.Adam(shared_policy.parameters(), lr=p_lr)
        v_optim = torch.optim.Adam(shared_value.parameters(), lr=v_lr)

        # compute gradients for local policy net
        self.policy.train()
        p_optim.zero_grad()
        pred_actions = self.policy(states)
        cost = policy_cost(pred_actions, actions, advantages, 0.01)
        cost.backward()
        # optimisation step on the shared policy net
        ensure_shared_grads(self.policy, shared_policy)
        p_optim.step()

        # compute gradients for local policy net
        self.value.train()
        v_optim.zero_grad()
        pred_values = self.value(states)
        cost = value_cost(pred_values, targets)
        cost.backward()
        # optimisation step on the shared policy net
        ensure_shared_grads(self.value, shared_value)
        v_optim.step()

    def copy(self, shared_policy, shared_value):
        self.policy.load_state_dict(shared_policy.state_dict())
        self.value.load_state_dict(shared_value.state_dict())

    def run(self, shared_policy, shared_value, p_lr, v_lr, max_steps, counter, lock):

        while True:
            # copy the global models
            self.copy(shared_policy, shared_value)

            # run n steps
            steps = self.run_n_steps(5, counter, lock)

            # check that we are still required to run
            if counter.value >= max_steps:
                break

            # update global model
            self.update(steps, shared_policy, shared_value, p_lr, v_lr)
