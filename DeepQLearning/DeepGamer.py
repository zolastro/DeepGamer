import numpy as np           # Handle matrices
from vizdoom import *        # Doom Environment
import tensorflow as tf

from agent import DQLAgent

import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames

from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs


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

### MODEL HYPERPARAMETERS
state_size = (84,84,4)      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot

### TRAINING HYPERPARAMETERS
total_episodes = 1000        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64             

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


if __name__ == "__main__":
    if training == True:
        agent = DQLAgent(state_size, action_size)
        # Init the game
        game.init()

        for e in range(total_episodes):
            # reset state in the beginning of each game
            # Set step to 0
            step = 0
            
            # Initialize the rewards of the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            # Remember that stack frame function also call our preprocess function.
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            while step < max_steps:
                step += 1
                action = possible_actions[agent.act(state)]
                # Do the action
                reward = game.make_action(action)
                episode_rewards.append(reward)

                # Look if the episode is finished
                done = game.is_episode_finished()

                
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros((84,84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    agent.remember(state, action, reward, next_state, done)
                    
                    # exit episode loop
                    step = max_steps

                    print("episode: {}/{}, score: {}"
                        .format(e, total_episodes, np.sum(episode_rewards)))
                else:
                    next_state = game.get_state().screen_buffer
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    agent.remember(state, action, reward, next_state, done)

                    state = next_state
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                    
            if (e % 10) == 0:
                agent.save("./model.h5")
                print("Model Saved")
    else:
        game, possible_actions = create_environment()
        totalScore = 0

        # Load the model
        agent = DQLAgent(state_size, action_size)
        agent.load('./model.h5')
        game.init()
        n_episodes = 100
        for i in range(n_episodes):
            
            done = False
            
            game.new_episode()
            
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
                
            while not game.is_episode_finished():
                # Take the biggest Q value (= the best action)
                choice = agent.predict(state.reshape((1, *state.shape)))
                
                # Take the biggest Q value (= the best action)
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