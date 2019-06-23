import numpy as np
import random
import tensorflow as tf

from replay_buffer import ReplayBuffer

from keras import layers
from keras import models
from keras.optimizers import Adam
from keras import backend as K


class DQN:
    def __init__(self, state_size, action_size, memory_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(memory_size)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_max = 1
        self.decay_step = 0
        self.epsilon_decay = 0.0001
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()


    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = models.Sequential()

        model.add(layers.Conv2D(32, (8, 8), strides=(4,4), activation='elu', input_shape=self.state_size))
        model.add(layers.Conv2D(64, (4,4), strides=(2,2), activation='elu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))

        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # Copy weights from model to target_model
        print('Updating target model...')
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, experience):
        self.memory.store_experience(experience)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, (1, 42, 42, self.state_size[2]))
        act_values = self.model.predict(state)

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = self.memory.get_experiences(batch_size)
        states, targets_f = [], []

        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, (1, 42, 42, self.state_size[2]))
            next_state = np.reshape(next_state, (1, 42, 42, self.state_size[2]))
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            states.append(state[0])
            targets_f.append(target[0])

        self.model.fit(np.array(states), np.array(targets_f), batch_size=batch_size, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * self.decay_step)
            self.decay_step += 1    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
