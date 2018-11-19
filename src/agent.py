import numpy as np
from collections import deque
import random

from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow.keras.backend as K


class Agent(object):
    def __init__(self, state_size, action_size, gamma=0.95, batch_size=128):
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=2000)
        print(self.state_size, self.action_size)

    def remember(self, state, action, reward, next_state):
        self.memory.append([state, action, reward, next_state])

    def _sample_minibatch(self):
        return random.sample(self.memory, min(len(self.memory), self.batch_size))

    def replay(self):
        raise NotImplementedError()

    def act(self, state):
        raise NotImplementedError()


class DumbAgent(Agent):
    def __init__(self, state_size, action_size, gamma=0.95,
                 batch_size=128):
        super().__init__(state_size, action_size,
                         gamma=gamma,
                         batch_size=batch_size)

    def remember(self, state, action, reward, next_state):
        pass

    def replay(self):
        pass

    def act(self, state):
        return 0


class RandomAgent(Agent):
    def __init__(self, state_size, action_size,
                 alpha=0.5, gamma=0.95,
                 batch_size=128):
        super().__init__(state_size, action_size,
                         gamma=gamma,
                         batch_size=batch_size)

    def remember(self, state, action, reward, next_state):
        pass

    def replay(self):
        pass

    def act(self, state):
        return np.random.randint(self.action_size)


class DNNAgent(Agent):
    def __init__(self, state_size, action_size,
                 alpha=0.5, gamma=0.95,
                 eta=1e-2, batch_size=128, hidden_dims=[64, 64]):
        super().__init__(state_size, action_size,
                         gamma=gamma,
                         batch_size=batch_size)
        self.hidden_dims = hidden_dims
        self.eta = eta
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError()

    def _get_dense_layers(self):
        ret = []
        for i in range(len(self.hidden_dims)):
            hidden_dim = self.hidden_dims[i]
            if i == 0:
                dense = Dense(hidden_dim,
                              input_dim=self.state_size,
                              activation='relu')
            else:
                dense = Dense(hidden_dim,
                              activation='relu')
            ret.append(dense)
        return ret

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class DQNAgent(DNNAgent):
    def __init__(self, state_size, action_size,
                 alpha=0.5, gamma=0.95,
                 eta=1e-2, batch_size=128):
        super().__init__(state_size, action_size,
                         gamma=gamma,
                         eta=eta, batch_size=batch_size)

        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def _build_model(self):
        model = Sequential(self._get_dense_layers() + [
            Dense(self.action_size, activation='linear'),
        ])

        model.compile(loss='mse',
                      optimizer=Adam(self.eta))

        model.summary()

        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = np.array([state])
        return np.argmax(self.model.predict(state))

    def replay(self):
        X = []
        Y = []
        minibatch = self._sample_minibatch()
        for s, a, r, s_ in minibatch:
            if s_ is None:
                target = r
            else:
                target = r + self.gamma * \
                    np.max(self.model.predict(np.array([s_])))
            y = self.model.predict(np.array([s]))[0]
            y[a] = target
            X.append(s)
            Y.append(y)
        X, Y = map(np.array, [X, Y])
        self.model.fit(X, Y, verbose=1, epochs=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PGAgent(DNNAgent):
    def __init__(self, state_size, action_size,
                 alpha=0.5, gamma=0.95,
                 eta=1e-3, batch_size=128):
        super().__init__(state_size, action_size,
                         gamma=gamma,
                         eta=eta, batch_size=batch_size)

    def _build_model(self):
        advantage = Input([1])

        policy_net = Sequential(self._get_dense_layers() + [
            Dense(self.action_size, activation='softmax'),
        ])

        def loss(y_true, y_pred):
            # y_true, the action
            # y_pred, the prob
            neg_log_prob = K.sparse_categorical_crossentropy(y_true, y_pred)
            # neg_log_prob = -K.sum(K.log(y_pred) * y_true, axis=1)
            return K.mean(advantage * neg_log_prob)

        model = Model(inputs=[policy_net.input,
                              advantage],
                      outputs=[policy_net.output])

        model.compile(loss=loss,
                      optimizer=RMSprop(self.eta))

        model.summary()

        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append([state, action, reward, next_state])

    def act(self, state):
        state = np.array([state])
        prob = self.model.predict([state, np.zeros(1)])[0]
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action

    def discount_rewards(self, reward):
        reward = reward.astype(np.float64)
        discounted_reward = np.zeros_like(reward)
        running_add = 0
        for t in reversed(range(0, reward.size)):
            running_add = running_add * self.gamma + reward[t]
            discounted_reward[t] = running_add
        return discounted_reward

    def replay(self):
        state, action, reward, _ = map(np.array, zip(*self.memory))
        reward = self.discount_rewards(reward)
        reward -= reward.mean()
        reward /= (reward.std() + 1e-10)
        self.model.fit([state, reward], action, verbose=1, epochs=1)
        self.memory = []
