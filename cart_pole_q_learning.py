import gym
import numpy as np
import pandas as pd


def select_action(state):
    return np.random.randint(0, 2)


def play_one_episode(env):
    # init variables
    curr_state = env.reset()
    done = False
    n_steps = 0

    # init containers
    ep_states = []
    ep_rewards = []

    # play episode
    while not done and n_steps < 10000:
        action = select_action(curr_state)
        curr_state, curr_reward, done, _ = env.step(action)
        ep_states.append(curr_state)
        ep_rewards.append(curr_reward)
        n_steps += 1

    return ep_states, ep_rewards


def sample_states(env, n_episodes):
    observed_states = []
    for _ in range(n_episodes):
        ep_states, _ = play_one_episode(env)
        observed_states += ep_states
    return np.array(observed_states)


def get_state_component_bins(obs_state_components, n_bins):
    min_bound = np.quantile(obs_state_components, 0.1)
    max_bound = np.quantile(obs_state_components, 0.9)
    bins = np.linspace(min_bound, max_bound, n_bins)
    return bins


def init_discrete_state(env, n_episodes, n_bins):
    observed_states = sample_states(env, n_episodes)
    state_ranges = []
    for state_component in range(4):
        obs_state_components = observed_states[:, state_component]
        bins = get_state_component_bins(obs_state_components, n_bins)
        state_ranges.append(bins)
    return state_ranges


class StateTransformer:

    def __init__(self, ranges):
        self.ranges = ranges

    def to_discrete(self, state):
        discrete_state = []
        for i in range(4):
            discrete_ind = np.digitize(state[i], self.ranges[i])
            discrete_state.append(discrete_ind)
        return discrete_state

    def to_string(self, discrete_state):
        return "".join([str(s) for s in discrete_state])

    def transform(self, state):
        discrete_state = self.to_discrete(state)
        string_state = self.to_string(discrete_state)
        return discrete_state, string_state


class QLearner:

    def __init__(self, transformer, gamma, learning_rate, eps):
        self.transformer = transformer
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps = eps
        self.Q = self._init_Q()

    def _init_action_space(self):
        return [i for i in range(2)]

    def _init_state_space(self):
        states = []
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    for p in range(10):
                        states.append(self.transformer.to_string([i, j, k, p]))
        return states

    def _init_Q(self):
        states = self._init_state_space()
        actions = self._init_action_space()
        return pd.DataFrame(np.random.uniform(low=-1, high=1, size=(len(states), len(actions))),
                            columns=actions,
                            index=states)

    def choose_action(self, state, train):
        if train and np.random.uniform(0, 1) < self.eps:
            return np.random.randint(0, 2)
        else:
            _, state_str = self.transformer.transform(state)
            action = self.Q.loc[state_str].argmax()
            return action

    def compute_target(self, reward, next_state):
        _, next_state_str = self.transformer.transform(next_state)
        target = reward + self.gamma * self.Q.loc[next_state_str].max()
        return target

    def update(self, state, action, target):
        _, state_str = self.transformer.transform(state)
        q_update = self.Q.loc[state_str, action] + self.learning_rate * (target - self.Q.loc[state_str, action])
        self.Q.loc[state_str, action] = q_update


# initialize
env = gym.make("CartPole-v0")
state_ranges = init_discrete_state(env, 1000, 9)
transformer = StateTransformer(state_ranges)
model = QLearner(transformer, gamma=0.9, learning_rate=0.1, eps=0.2)

# main
total_rewards = np.empty(2000)

for i in range(2000):

    state = env.reset()
    done = False
    iter = 0
    eps = 1 / np.sqrt(i+1)
    model.eps = eps
    total_reward = 0

    while not done and iter < 10000:
        action = model.choose_action(state, train=True)
        next_state, reward, done, _ = env.step(action)
        if iter < 199 and done:
            reward = -300
        total_reward += reward
        target = model.compute_target(reward, next_state)
        model.update(state, action, target)
        state = next_state
        iter += 1

    total_rewards[i] = total_reward
    if i % 100 == 0:
        print(f"avg reward at {i}: {total_rewards[i-100:i].mean()}")

# try one run
state = env.reset()
done = False
length = 0
while not done:
    action = model.choose_action(state, train=False)
    state, _, done, _ = env.step(action)
    env.render()
    length += 1

print("length", length)
