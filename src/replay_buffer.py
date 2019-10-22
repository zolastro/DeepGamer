from collections import deque
import random
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def get_experiences(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def store_experience(self, experience):
        self.buffer.append(experience)
    
    def get_length(self):
        return len(self.buffer)