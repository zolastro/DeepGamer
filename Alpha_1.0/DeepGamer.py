import numpy as np           # Handle matrices
from vizdoom import *        # Doom Environment
import tensorflow as tf

import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames

from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

from keras import layers
from keras import models
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import TensorBoard

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

import datetime

now = datetime.datetime.now();

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
       
"""
Here we performing random action to test the environment
"""
def test_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print ("\treward:", reward)
            time.sleep(0.02)
        print ("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()


def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)
    
    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10,30:-30]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    
    return preprocessed_frame



stack_size = 4 # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4) 

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    frame = np.expand_dims(frame, axis=0)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=3)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=3) 
    
    stacked_state = stacked_state.reshape((84,84,4))
    return stacked_state, stacked_frames


game, possible_actions = create_environment()
# Run tensorboard with 
tensorboard = TensorBoard(log_dir="./logs", histogram_freq=0,write_graph=True, write_images=False)

### MODEL HYPERPARAMETERS
state_size = (84,84,4)      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot
learning_rate =  0.0002      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 500        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64             

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


model = models.Sequential()
model.add(layers.Conv2D(32, (8, 8), strides=(4,4), activation='elu', input_shape=state_size))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(layers.Conv2D(64, (4,4), strides=(2,2), activation='elu', input_shape=state_size))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), activation='elu', input_shape=state_size))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(action_size, activation='linear'))

model.compile(optimizer=RMSprop(lr=learning_rate), 
    loss='mean_squared_error', 
    metrics=['acc'])

# model.compile(optimizer=RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=decay_rate), 
# loss='mean_squared_error', 
# metrics=['acc'])

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]

# Instantiate memory
memory = Memory(max_size = memory_size)

# Render the environment
game.new_episode()

for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # First we need a state
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    
    # Random action
    action = random.choice(possible_actions)
    
    # Get the rewards
    reward = game.make_action(action)
    
    # Look if the episode is finished
    done = game.is_episode_finished()
    
    # If we're dead
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)
        
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        
        # Start a new episode
        game.new_episode()
        
        # First we need a state
        state = game.get_state().screen_buffer
        
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    else:
        # Get the next state
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        
        # Our state is now the next_state
        state = next_state



def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()
    state = state.reshape((1,84,84,4))
    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = model.predict(state)
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
    return action, explore_probability

# Saver will help us to save our model

if training == True:    
    # Initialize the decay rate (that will use to reduce epsilon) 
    decay_step = 0

    # Init the game
    game.init()

    for episode in range(total_episodes):
        # Set step to 0
        step = 0
        
        # Initialize the rewards of the episode
        episode_rewards = []
        
        # Make a new episode and observe the first state
        game.new_episode()
        state = game.get_state().screen_buffer
        
        # Remember that stack frame function also call our preprocess function.
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        while step < max_steps:
            step += 1
            
            # Increase decay_step
            decay_step +=1
            
            # Predict the action to take and take it
            action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

            # Do the action
            reward = game.make_action(action)

            # Look if the episode is finished
            done = game.is_episode_finished()
            
            # Add the reward to total reward
            episode_rewards.append(reward)

            # If the game is finished
            if done:
                # the episode ends so no next state
                next_state = np.zeros((84,84), dtype=np.int)
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                # Set step = max_steps to end the episode
                step = max_steps

                # Get the total reward of the episode
                total_reward = np.sum(episode_rewards)


                memory.add((state, action, reward, next_state, done))

            else:
                # Get the next state
                next_state = game.get_state().screen_buffer
                
                # Stack the frame of the next_state
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                

                # Add experience to memory
                memory.add((state, action, reward, next_state, done))
                
                # st+1 is now our current state
                state = next_state


            ### LEARNING PART            
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=4)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch]) 
            next_states_mb = np.array([each[3] for each in batch], ndmin=4)
            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []

            # Get Q values for next_state
            Qs_next_state = model.predict(next_states_mb)
            
            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
            for i in range(0, len(batch)):
                terminal = dones_mb[i]

                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards_mb[i])
                    
                else:
                    target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)
                    

            targets_mb = np.array([each for each in target_Qs_batch])

            model.fit(x=states_mb, y=actions_mb, batch_size=64, epochs=1, verbose=2, callbacks=[tensorboard])

            
            # loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
            #                     feed_dict={DQNetwork.inputs_: states_mb,
            #                                 DQNetwork.target_Q:   ,
            #                                 DQNetwork.actions_: actions_mb})

            # # Write TF Summaries
            # summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
            #                                     DQNetwork.target_Q: targets_mb,
            #                                     DQNetwork.actions_: actions_mb})
            # writer.add_summary(summary, episode)
            # writer.flush()

        # Save model every 5 episodes
        if episode % 5 == 0:
            model.save("./model.h5")
            print("Model Saved")
else:
    game, possible_actions = create_environment()
    totalScore = 0
    
    # Load the model
    model = load_model('./model.h5')
    game.init()
    n_episodes = 100
    for i in range(n_episodes):
        
        done = False
        
        game.new_episode()
        
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
            
        while not game.is_episode_finished():
            # Take the biggest Q value (= the best action)
            Qs = model.predict(state.reshape((1, *state.shape)))
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]
            
            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()
            totalScore += score
            
            if done:
                break  
                
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state

        score = game.get_total_reward()
        print("Score: ", score)
    print ("Mean Score")
    print (totalScore/n_episodes)
    game.close()