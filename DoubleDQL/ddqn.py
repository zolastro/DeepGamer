import random
import gym
import numpy as np
from collections import deque
from keras import layers
from keras import models
from keras.optimizers import Adam
from keras import backend as K

from vizdoom import *        # Doom Environment
from skimage import transform# Help us to preprocess the frames

from PIL import Image, ImageEnhance 

from matplotlib import pyplot as plt

import tensorflow as tf

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=150000)
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

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

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
        # copy weights from model to target_model
        print('Updating target model...')
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, (1, 42, 42, stack_size))
        act_values = self.model.predict(state)
        if np.random.rand() < 0.01:
            print(act_values)

        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []

        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, (1, 42, 42, stack_size))
            next_state = np.reshape(next_state, (1, 42, 42, stack_size))
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                #print('Target: {}, Reward: {}, Action: {}, Target_f: {}'.format(t, reward, action, target))
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
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

def create_environment():
    game = DoomGame()
    
    # Load the correct configuration
    game.load_config("basic.cfg")
    
    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("basic.wad")
    
    game.init()
    
    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    
    return game, possible_actions
       
def increase_contrast(img, factor):
    factor = float(factor)
    return np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8)


def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)
    
    # Crop the screen (remove the roof because it contains no information)
    frame = increase_contrast(frame, 10)

    cropped_frame = frame[30:-40,30:-30]

    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [42,42])
    
    
    return preprocessed_frame



stack_size = 2 # We stack 4 frames
frame_wh = 42
# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((42,42), dtype=np.int) for i in range(stack_size)], maxlen=stack_size) 

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    frame = np.expand_dims(frame, axis=0)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((42,42), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=3)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=3) 
    
    stacked_state = stacked_state.reshape((42,42,stack_size))
    return stacked_state, stacked_frames


game, possible_actions = create_environment()

### MODEL HYPERPARAMETERS
state_size = (42,42,stack_size)      # Our input is a stack of 4 frames hence 42x42x2 (Width, height, channels)
action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot

### TRAINING HYPERPARAMETERS
EPISODES = 5000        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 256             
min_replay_size = 5000
### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

if __name__ == "__main__":
    # Init the game
    game.init()
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    e = 0

    while True:
        e += 1
        game.new_episode()
        # Make a new episode and observe the first state
        # Remember that stack frame function also call our preprocess function.
        game.new_episode()
        episode_rewards = []

        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        for time in range(100):
            # env.render()
            action = agent.act(state)
            reward = game.make_action(possible_actions[action])
            done = game.is_episode_finished()

            reward = reward if not done else -10
            episode_rewards.append(reward)

            if done or time == 99:
                agent.update_target_model()
                print("episode: {}, score: {}, e: {:.2}"
                    .format(e, np.sum(episode_rewards), agent.epsilon))


            if done:
                break
                # exit episode loop
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) >= min_replay_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            print('Agent saved')
            print('Memory: {}'.format(len(agent.memory)))
            agent.save("./save/ddqn.h5")
