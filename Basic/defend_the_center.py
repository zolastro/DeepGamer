import random
import numpy as np

from dqn import DQN
from collections import deque
from environment import Environment
from skimage import transform# Help us to preprocess the frames


from PIL import Image, ImageEnhance 


### FRAME PREPROCESSING FUNCTIONS
def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # Increase contrast
    frame = increase_contrast(frame, 4)
    # Remove the roof because it contains no information)
    cropped_frame = frame[30:-40,30:-30]

    # Normalize pixel values to be between 0.0 and 1.0
    normalized_frame = cropped_frame/255.0
    
    # Resize the preprocessed frame down to 42x42 
    preprocessed_frame = transform.resize(normalized_frame, [42,42])
    
    return preprocessed_frame

def increase_contrast(img, factor):
    factor = float(factor)
    return np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8)

### FRAME STACKING FUNCTION
def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    frame = np.expand_dims(frame, axis=0)

    if is_new_episode:
        # Clear our previously stacked frames
        stacked_frames = deque([np.zeros((42,42), dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
        # Initialize the frames deque by copying the same frame "stack_size" times
        for _ in range(stack_size):
            stacked_frames.append(frame)
    else:
        # Append the frames to the deque
        stacked_frames.append(frame)
    
    # Create an stack out of the deque
    stacked_state = np.stack(stacked_frames, axis=3) 
    stacked_state = stacked_state.reshape((42,42,stack_size))
    return stacked_state, stacked_frames

### ENVIRONMENT HYPERPARAMETERS
left = [1, 0, 0]
right = [0, 1, 0]
shoot = [0, 0, 1]
possible_actions = [left, right, shoot]
game = Environment("defend_the_center/defend_the_center.cfg", "defend_the_center/defend_the_center.wad")

### EXPERIENCES HYPERPARAMETERS
stack_size = 2                                      # We stack 2 frames per experience
# Creates a first stack of frames initialized to zeros
stacked_frames = deque([np.zeros((42,42), dtype=np.int) for i in range(stack_size)], maxlen=stack_size) 

### MODEL HYPERPARAMETERS
state_size = (42,42,stack_size)                     # Our input is a stack of 4 frames hence 42x42x2 (Width, height, channels)
action_size = game.get_available_buttons_size()     # 3 possible actions: left, right, shoot

### TRAINING HYPERPARAMETERS
max_steps = 500                                     # Max possible steps in an episode
batch_size = 32             
min_replay_size = 5000                              # Minimum number of experiences stored before start training the DQN


if __name__ == "__main__":
    # Create a DQN with a replay buffer capacity up to 150000 experiences
    agent = DQN(state_size, action_size, 150000)
    # Initialize episode counter
    e = 0
    while True:
        # Make a new episode
        game.new_episode()
        episode_rewards = []

        # Get the current environment state and add it to the previously stacked frames
        state = game.get_state()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        for time in range(max_steps):
            # Get next action from the DQN
            action = agent.act(state)
            # Perform that action and recieve its reward
            reward = game.make_action(possible_actions[action])
            episode_rewards.append(reward)
            # Check whether the episode is finished or not
            done = game.is_episode_finished() or time == max_steps

            if done:
                # Episode finished
                agent.update_target_model()
                print("Episode: {}, score: {}, e: {:.2}"
                    .format(e, np.sum(episode_rewards), agent.epsilon))
                # exit episode loop
                break
            else:
                # Get the next environment state and stack it to the previously stacked frames
                next_state = game.get_state()
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            
            # Add that experience to the replay buffer
            agent.remember((state, action, reward, next_state, done))

            # Set the next state to the current one
            state = next_state

            # If the replay buffer already has "min_replay_size" experiences, train the DQN
            if agent.memory.get_length() >= min_replay_size:
                agent.replay(batch_size)
                e += 1
        # Save the DQN weights every 10 episodes
        if e % 10 == 0:
            print('Agent saved')
            print('Replay buffer length: {}'.format(agent.memory.get_length()))
            agent.save("./save/defend_the_center.h5")