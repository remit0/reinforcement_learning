import gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler


class StateTransformer:

    """ Standardize and applies different RBF kernels to state vectors.
    Calibrated on randomly generated state vectors. """

    def __init__(self, env, n_rbf_comp=500, n_exemplars=10000):
        self.env = env
        self.n_rbf_comp = n_rbf_comp
        self.n_exemplars = n_exemplars
        self.pipe = self._build_pipeline()
        self.dimension = self._init_pipe()

    def _get_exemplars(self):
        exemplars = np.array([self.env.observation_space.sample() for _ in range(self.n_exemplars)])
        return exemplars

    def _build_pipeline(self):
        featurizer = FeatureUnion([
            ("RBF1", RBFSampler(gamma=5.0, n_components=self.n_rbf_comp)),
            ("RBF2", RBFSampler(gamma=2.0, n_components=self.n_rbf_comp)),
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


class Model:

    """ QLearner with linear models. """

    def __init__(self, env, transformer, learning_rate):
        self.env = env
        self.transformer = transformer
        self.learning_rate = learning_rate
        self.action_models = self._init_action_models()

    def _init_action_models(self):
        action_models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate=self.learning_rate)
            # dummy fit -- mandatory from sklearn's API
            model.partial_fit(self.transformer.transform([env.reset()]), [0])
            action_models.append(model)
        return action_models

    def predict(self, state):
        scaled = self.transformer.transform([state])
        result = np.stack([m.predict(scaled) for m in self.action_models]).T[0]
        return result

    def update(self, state, action, target):
        scaled = self.transformer.transform([state])
        self.action_models[action].partial_fit(scaled, [target])

    def choose_action(self, state, eps):
        if np.random.uniform(0, 1) < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(state))

    def compute_target(self, reward, next_state, gamma):
        target = reward + gamma * np.max(self.predict(next_state))
        return target


def train_one_episode(model, env, eps, gamma):
    state = env.reset()
    done = False
    episode_reward = 0
    n_steps = 0
    while not done and n_steps < 10000:
        action = model.choose_action(state, eps=eps)
        next_state, reward, done, _ = env.step(action)
        target = model.compute_target(reward, next_state, gamma)
        model.update(state, action, target)
        state = next_state
        n_steps += 1
        episode_reward += reward
    return episode_reward


def play_one_episode(model, env):
    env = gym.wrappers.Monitor(env, 'output', force=True)
    state = env.reset()
    done = False
    while not done:
        action = model.choose_action(state, eps=eps)
        next_state, reward, done, _ = env.step(action)
        state = next_state


env = gym.make("MountainCar-v0").env
transformer = StateTransformer(env=env, n_rbf_comp=10)
model = Model(env, transformer, learning_rate="constant")
rewards = []

for i in range(150):
    eps = 0.1 * (0.97**i)
    episode_reward = train_one_episode(model, env, eps, gamma=0.99)
    rewards.append(episode_reward)
    if i % 10 == 0:
        print(f"avg reward at episode {i}: {np.mean(rewards[-100:])}")


play_one_episode(model, env)
