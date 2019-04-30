from collections import deque

import numpy as np
import sys

import random as random

from keras import layers
from keras import models
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt # Display graphs

import pickle 


class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.6    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.decay_step = 0
        self.epsilon_decay = 0.0001
        self.learning_rate = 0.0002
        self.rho=0.95
        self.model = self._build_model()
        self.tensorboard = TensorBoard(log_dir="/home/zolastro/Documents/DeepGamer/DeepQLearning/logs", histogram_freq=0,write_graph=True, write_images=False)


    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=(4,4), activation='elu', input_shape=self.state_size))
        model.add(BatchNormalization())
        model.add(layers.Conv2D(64, (4,4), strides=(2,2), activation='elu'))
        model.add(BatchNormalization())
        model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), activation='elu'))
        model.add(BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))

        model.compile(optimizer=RMSprop(lr=self.learning_rate, rho=self.rho), 
            loss='mse', 
            metrics=['acc'])
        model.summary()
        return model

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if (len(self.memory) % 1000):
            memory_file = open('memory.obj', 'wb') 
            pickle.dump(self.memory, memory_file)
            memory_file.close()


    def plot_state(self, state):
        images = state
        images = np.swapaxes(images, 2, 0)
        images = np.swapaxes(images, 2, 1)

        plt.imshow(images[0], interpolation='nearest')
        plt.show()

    def summary(self):
        print('Epsilon:')
        print(self.epsilon)
        print('Memory:')
        print(len(self.memory))

    def act(self, state):
        state = np.reshape(state, (1, 84, 84, 4))
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * self.decay_step)
        self.decay_step += 1

        if self.epsilon > np.random.rand() :
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def predict(self, state):
        act_values = self.model.predict(state)
        print(act_values)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, (1, 84, 84, 4))
            next_state = np.reshape(next_state,  (1, 84, 84, 4))
            action = np.argmax(action)
            if done:
                target = reward 
            else:
                prediction = self.model.predict(next_state)[0]
            target_f = self.model.predict(state)
            target_f[0][action] = target
            print(target_f[0])
            print('------------')
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        # print(np.mean(np.swapaxes(np.array(targets_f), 0, 1), axis=1))
        self.model.fit(np.array(states), np.array(targets_f), epochs=1, batch_size=batch_size, verbose=0, callbacks=[self.tensorboard])
