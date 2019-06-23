import random
import numpy as np

from dqn import DQN
from collections import deque
from environment import Environment
from skimage import transform# Help us to preprocess the frames


from PIL import Image, ImageEnhance 



def create_environment():
    game = Environment("basic.cfg", "basic.wad")

    
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
    # Crop the screen (remove the roof because it contains no information)
    frame = increase_contrast(frame, 4)

    cropped_frame = frame[30:-40,30:-30]

    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [42,42])
    
    return preprocessed_frame



stack_size = 2 # We stack 4 frames
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
max_steps = 1000       # Max possible steps in an episode
batch_size = 32             
min_replay_size = 5000
### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

if __name__ == "__main__":
    # Init the game
    agent = DQN(state_size, action_size, 150000)
    done = False
    e = 1
    all_scores = []
    while True:
        if agent.memory.get_length() >= min_replay_size:
            e += 1        # Make a new episode and observe the first state
        # Remember that stack frame function also call our preprocess function.
        game.new_episode()
        episode_rewards = []

        state = game.get_state()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        for time in range(max_steps):
            # env.render()
            action = agent.act(state)
            reward = game.make_action(possible_actions[action])
            done = game.is_episode_finished()

            reward = reward if not done else 1
            episode_rewards.append(reward)

            if done or time == max_steps:
                agent.update_target_model()
                print("episode: {}, score: {}, e: {:.2}"
                    .format(e, np.sum(episode_rewards), agent.epsilon))
                all_scores.append(np.sum(episode_rewards))

            if done:
                break
                # exit episode loop
            else:
                next_state = game.get_state()
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            
            agent.remember((state, action, reward, next_state, done))
            state = next_state
            if agent.memory.get_length() >= min_replay_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            print('Agent saved')
            print('Memory: {}'.format(agent.memory.get_length()))
            agent.save("./save/ddqn.h5")