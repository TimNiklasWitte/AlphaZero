import gym
import tensorflow as tf
import numpy as np

import tqdm

from MCTS import *

class ReplayMemory:

    def __init__(self, max_size) -> None:
        self.max_size = max_size

        self.state_list = np.zeros(shape=(max_size, 4), dtype=np.float32)
        self.policy_list = np.zeros(shape=(max_size, 2), dtype=np.float32)
        self.value_list = np.zeros(shape=(max_size, 1), dtype=np.float32)

        self.write_idx = 0
        self.was_full = False

    def add_sample(self, state, policy, value):
        
        self.state_list[self.write_idx] = state
        self.policy_list[self.write_idx] = policy
        self.value_list[self.write_idx] = value

        self.write_idx += 1

        if self.write_idx == self.max_size:
            self.write_idx = 0
            
            self.was_full = True

    def dataset_generator(self):
        
        if self.was_full:
            size = self.max_size
        else:
            size = self.write_idx

       
        for idx in range(size):
            state = self.state_list[idx]
            policy = self.policy_list[idx]
            value = self.value_list[idx]

            yield state, policy, value
        
 
            

SHUFFLE_WINDOW_SIZE = 2000
BATCH_SIZE = 32
def prepare_data(dataset):
    dataset = dataset.shuffle(SHUFFLE_WINDOW_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
