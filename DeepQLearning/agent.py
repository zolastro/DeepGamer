from collections import deque

import numpy as np

import random as random

from keras import layers
from keras import models
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import TensorBoard


class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0002
        self.model = self._build_model()
        self.tensorboard = TensorBoard(log_dir="/tmp/tensorboard/logs", histogram_freq=0,write_graph=True, write_images=False)


    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=(4,4), activation='elu', input_shape=self.state_size))
        model.add(BatchNormalization())
        model.add(layers.Conv2D(64, (4,4), strides=(2,2), activation='elu'))
        model.add(BatchNormalization())
        model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), activation='elu'))
        model.add(BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(self.action_size, activation='linear'))

        model.compile(optimizer=RMSprop(lr=self.learning_rate), 
            loss='mse', 
            metrics=['acc'])

        return model

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model('./model.h5')


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, (1, 84, 84, 4))

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def predict(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, (1, 84, 84, 4))
            next_state = np.reshape(next_state,  (1, 84, 84, 4))
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0, callbacks=[self.tensorboard])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
