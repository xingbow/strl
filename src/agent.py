from keras import Sequential, Model
from keras.optimizers import Adam
from collections import deque

from keras.layers import Input, Dense, Conv2D, Activation, Lambda, concatenate
from keras.utils import np_utils


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


class VanillaPGAgent(object):

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self._build_model(state_size, action_size)

    def _build_model(self, state_size, action_size):
        model = Sequential()
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop')

        self.model = model

    def act(self, state):
        """Returns an action at given `state`

        Args:
            state (1-D or 2-D Array): It can be either 1-D array of shape (state_dimension, )
                or 2-D array shape of (n_samples, state_dimension)

        Returns:
            action: an integer action value ranging from 0 to (n_actions - 1)
        """
        shape = state.shape

        if len(shape) == 1:
            assert shape == (self.state_size,), "{} != {}".format(
                shape, self.state_size)
            state = np.expand_dims(state, axis=0)

        elif len(shape) == 2:
            assert shape[1] == (self.state_size), "{} != {}".format(
                shape, self.state_size)

        else:
            raise TypeError(
                "Wrong state shape is given: {}".format(state.shape))

        action_prob = np.squeeze(self.model.predict(state))
        assert len(action_prob) == self.action_size, "{} != {}".format(
            len(action_prob), self.action_size)
        return np.random.choice(np.arange(self.action_size), p=action_prob)

    def replay(self, S, A, R):
        """Train a network

        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.

        """
        action_onehot = np_utils.to_categorical(
            A, num_classes=self.action_size)

        discounted_rewards = self._discounted_R(R)
        advantage = (discounted_rewards - discounted_rewards.mean()
                     ) / discounted_rewards.std()

        self.model.fit(S, action_onehot, sample_weight=advantage, nb_epoch=1)

    def _discounted_R(self, R, discount_rate=.99):
        discounted_r = np.zeros_like(R, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(R))):
            running_add = running_add * discount_rate + R[t]
            discounted_r[t] = running_add
        return discounted_r
