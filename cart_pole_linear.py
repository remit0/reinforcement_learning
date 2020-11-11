import gym
import numpy as np

N_RUNS = 5
N_EPISODES = 5


def generate_weights(size):
    weights = np.random.rand(size)
    weights = weights / np.linalg.norm(weights)
    return weights


def get_action(state, weights):
    action = 1 if state.dot(weights) > 0 else 0
    return action


def play_one_episode(env, weights):
    state = env.reset()
    done = False
    length = 0
    while not done and length < 10000:
        action = get_action(state, weights)
        state, _, done, _ = env.step(action)
        length += 1
    return length


def play_n_episodes(env, weights, n):
    avg_eps_len = 0
    for _ in range(n):
        eps_len = play_one_episode(env, weights)
        avg_eps_len += eps_len
    avg_eps_len /= n
    return avg_eps_len


env = gym.make("CartPole-v0")

best_avg_eps_len = 0
for _ in range(N_RUNS):
    new_weights = generate_weights(4)
    avg_eps_len = play_n_episodes(env, new_weights, N_EPISODES)
    if avg_eps_len > best_avg_eps_len:
        best_avg_eps_len = avg_eps_len
        weights = new_weights

# print results
print(best_avg_eps_len)
print(weights)

# play a final set of episodes
env = gym.wrappers.Monitor(env, "output")
play_one_episode(env, weights)
