import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.capacity:
            del self.buffer[:-self.capacity]

    def extend(self, items):
        self.buffer.extend(items)
        if len(self.buffer) > self.capacity:
            del self.buffer[:-self.capacity]

    def sample(self, count):
        return [self.buffer[i] for i in np.random.choice(np.arange(len(self.buffer)), count)]
