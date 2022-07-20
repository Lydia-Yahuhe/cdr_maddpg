from collections import namedtuple
import random

Experience = namedtuple('Experience', ('states', 'actions', 'next_states', 'rewards'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = {}
        self.position = {}

    def push(self, *args):
        key = len(args[0])
        if key not in self.memory.keys():
            memory = self.memory[key] = []
            position = self.position[key] = 0
        else:
            memory = self.memory[key]
            position = self.position[key]

        if len(memory) < self.capacity:
            memory.append(None)

        memory[position] = Experience(*args)
        self.position[key] = int((position + 1) % self.capacity)

    def sample(self, batch_size):
        samples = {}
        for n, memory in self.memory.items():
            # print(n, len(memory))
            if len(memory) >= batch_size:
                samples[n] = random.sample(memory, batch_size)
        return samples

    def __len__(self):
        return len(self.memory)
