import numpy as np

from collections import deque
from skimage import transform
from PIL import Image, ImageEnhance 

class Preprocessor:
    def __init__(self, stack_size):
        self.stack_size = stack_size

    def _preprocess_frame(self, frame):
        # Greyscale frame already done in our vizdoom config
        # Increase contrast
        frame = self._increase_contrast(frame, 4)
        # Remove the roof because it contains no information)
        cropped_frame = frame[30:-40,30:-30]

        # Normalize pixel values to be between 0.0 and 1.0
        normalized_frame = cropped_frame/255.0
        
        # Resize the preprocessed frame down to 42x42 
        preprocessed_frame = transform.resize(normalized_frame, [42,42])
        
        return preprocessed_frame

    def _increase_contrast(self, img, factor):
        factor = float(factor)
        return np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8)

    def stack_frames(self, stacked_frames, state, is_new_episode):
        # Preprocess frame
        frame = self._preprocess_frame(state)
        frame = np.expand_dims(frame, axis=0)

        if is_new_episode:
            # Clear our previously stacked frames
            stacked_frames = deque([np.zeros((42,42), dtype=np.int) for i in range(self.stack_size)], maxlen=self.stack_size)
            # Initialize the frames deque by copying the same frame "stack_size" times
            for _ in range(self.stack_size):
                stacked_frames.append(frame)
        else:
            # Append the frames to the deque
            stacked_frames.append(frame)
        
        # Create an stack out of the deque
        stacked_state = np.stack(stacked_frames, axis=3) 
        stacked_state = stacked_state.reshape((42,42,self.stack_size))
        return stacked_state, stacked_frames