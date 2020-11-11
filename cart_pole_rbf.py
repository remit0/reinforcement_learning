import gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


class StateTransformer:

    """ Standardize and applies different RBF kernels to state vectors.
    Calibrated on randomly generated state vectors. """

    def __init__(self, env, n_init_samples=1000, n_rbf_comp=500, n_exemplars=10000):
        self.env = env
        self.n_init_samples = n_init_samples
        self.n_rbf_comp = n_rbf_comp
        self.n_exemplars = n_exemplars
        self.pipe = self._build_pipeline()
        self.dimension = self._init_pipe()

    def _get_episode_states(self):
        ep_states = []
        self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()
            state, _, done, _ = env.step(action)
            ep_states.append(state)
        return ep_states

    def _get_sample_states(self):
        states = []
        for _ in range(self.n_init_samples):
            ep_states = self._get_episode_states()
            states += ep_states
        return np.array(states)

    def _get_exemplars(self):
        states = self._get_sample_states()
        indices = np.random.randint(len(states), size=int(self.n_init_samples / 10))
        exemplars = states[indices, :]
        return exemplars

    def _build_pipeline(self):
        featurizer = FeatureUnion([
            ("RBF1", RBFSampler(gamma=0.05, n_components=self.n_rbf_comp)),
            ("RBF2", RBFSampler(gamma=0.1, n_components=self.n_rbf_comp)),
            ("RBF3", RBFSampler(gamma=1.0, n_components=self.n_rbf_comp)),
            ("RBF4", RBFSampler(gamma=0.5, n_components=self.n_rbf_comp)),
        ])
        pipe = Pipeline(
            steps=[
                ("StandardScaler", StandardScaler()),
                ("RBF", featurizer)
            ]
        )
        return pipe

    def _init_pipe(self):
        exemplars = self._get_exemplars()
        features = self.pipe.fit_transform(exemplars)
        return features.shape[1]

    def transform(self, states):
        scaled = self.pipe.transform(states)
        return scaled


class SGDRegressor:

    def __init__(self, learning_rate, dimension):
        self.learning_rate = learning_rate
        self.W = self._init_weights(dimension)

    def _init_weights(self, dimension):
        return np.random.randn(dimension) / np.sqrt(dimension)

    def partial_fit(self, x, y, e):
        error = y - self.predict(x)
        self.W = self.W + self.learning_rate * error * e

    def predict(self, x):
        return x[0].dot(self.W)


class Model:

    """ QLearner with linear models. """

    def __init__(self, env, transformer, learning_rate):
        self.env = env
        self.transformer = transformer
        self.learning_rate = learning_rate
        self.models = self._init_models()
        self.eligibilities = np.zeros((env.action_space.n, transformer.dimension))

    def _init_models(self):
        models = [SGDRegressor(learning_rate=self.learning_rate, dimension=self.transformer.dimension)
                  for _ in range(env.action_space.n)]
        return models

    def predict(self, state):
        scaled = self.transformer.transform([state])
        result = [m.predict(scaled) for m in self.models]
        return result

    def update(self, state, action, target, gamma, lda):
        scaled = self.transformer.transform([state])
        self.eligibilities *= gamma * lda
        self.eligibilities[action] += scaled[0]
        self.models[action].partial_fit(scaled, target, self.eligibilities[action])

    def choose_action(self, state, eps):
        if np.random.uniform(0, 1) < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(state))

    def compute_target(self, reward, next_state, gamma):
        target = reward + gamma * np.max(self.predict(next_state))
        return target


def train_one_episode(model, env, eps, gamma, lda):
    # initialization
    state, done, episode_reward, n_steps = env.reset(), False, 0, 0
    while not done and n_steps < 10000:
        action = model.choose_action(state, eps=eps)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -200

        target = model.compute_target(reward, next_state, gamma)
        model.update(state, action, target, gamma, lda)
        state = next_state
        n_steps += 1
        episode_reward += reward
    return episode_reward


def play_one_episode(model, env):
    env = gym.wrappers.Monitor(env, 'output', force=True)
    state = env.reset()
    done = False
    steps = 0
    while not done:
        action = model.choose_action(state, eps=-1)
        state, _, done, _ = env.step(action)
        steps += 1
    print("number of steps", steps)


env = gym.make("CartPole-v0").env
transformer = StateTransformer(env=env)
model = Model(env, transformer, learning_rate=1e-2)
rewards = []

for i in range(200):
    eps = 0.1 * (0.97**i)
    episode_reward = train_one_episode(model, env, eps, gamma=0.9, lda=1)
    rewards.append(episode_reward)
    if i % 10 == 0:
        print(f"avg reward at episode {i}: {np.mean(rewards[-100:])}")


play_one_episode(model, env)
