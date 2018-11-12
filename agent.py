from keras import Sequential, Model
from keras.optimizers import Adam
from collections import deque

from keras.layers import Input, Dense, Conv2D, Activation, Lambda, concatenate

import numpy as np
import random


class DQNAgent(object):
    def __init__(self, num_regions, action_size):
        self.num_regions = num_regions
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        n = self.num_regions

        input_ = Input([6 * n + 4])

        a0 = Lambda(lambda x: x[:, 0:n])(input_)
        s0 = Lambda(lambda x: x[:, n:2 * n])(input_)
        b0 = Lambda(lambda x: x[:, 2 * n:2 * n + 1])(input_)
        s0_ = Lambda(lambda x: x[:, 2 * n + 1:3 * n + 1])(input_)
        b0_ = Lambda(lambda x: x[:, 3 * n + 1:3 * n + 2])(input_)

        a1 = Lambda(lambda x: x[:, 3 * n + 2:4 * n + 2])(input_)
        s1 = Lambda(lambda x: x[:, 4 * n + 2:5 * n + 2])(input_)
        b1 = Lambda(lambda x: x[:, 5 * n + 2:5 * n + 3])(input_)
        s1_ = Lambda(lambda x: x[:, 5 * n + 3:6 * n + 3])(input_)
        b1_ = Lambda(lambda x: x[:, 6 * n + 3:6 * n + 4])(input_)

        embedding_layer = Dense(128, activation='relu')

        embedding = concatenate([embedding_layer(x)for x in [s0, s0_, s1, s1_]]
                                + [a0, b0, b0_, a1, b1, b1_])

        fully_connect_layers = Sequential([
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])

        output = fully_connect_layers(embedding)

        model = Model(inputs=[input_], outputs=[output])

        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        # print(model.summary())

        return model

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, -1])
        next_state = np.reshape(state, [1, -1])
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, -1])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class VanillaDQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, -1])
        next_state = np.reshape(state, [1, -1])
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, -1])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
