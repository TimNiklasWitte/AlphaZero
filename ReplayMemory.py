import gym
import tensorflow as tf
import numpy as np

import tqdm

from MCTS import *

class ReplayMemory:

    def __init__(self, size) -> None:
        self.size = size

        self.state_list = []
        self.policy_list = []
        self.value_list = []

        # self.state_list = np.zeros(shape=(size, 4))
        # self.policy_list = np.zeros(shape=(size, 2))
        # self.value_list = np.zeros(shape=(size, 1))


    def add_sample(self, state, policy, value):
        self.state_list.append(state) 
        self.policy_list.append(policy)
        self.value_list.append(value)


    def fill(self, policyValueNetwork):

        env = gym.make("CartPole-v1") 
        state, _ = env.reset()

        eval_num_steps_list = []

        state_list = []
        policy_list = []
        reward_list = []

        done = False
        max_steps = 350
        cnt_steps = 0

        gamma = 0.9
        discount_factors = np.array([gamma**i for i in range(max_steps + 1)])

        for i in tqdm.tqdm(range(self.size),position=0, leave=True):
            mcts = MCTS(env, state, policyValueNetwork)

            policy = mcts.run(100)

            state_list.append(state)
            policy_list.append(policy)
        
            action = np.random.choice(2, p=policy)
            state, reward, done, _, _ = env.step(action)
            reward_list.append(reward)

            cnt_steps += 1
             
            if done or cnt_steps == max_steps:
                eval_num_steps_list.append(cnt_steps)
                
                state, _ = env.reset()

                for idx, (state, policy) in enumerate(zip(state_list, policy_list)):
                    value = np.dot(reward_list[idx:], discount_factors[:cnt_steps-idx])
                    value = np.array([value])
                    self.add_sample(state, policy, value)
                 
                cnt_steps = 0
                state_list = []
                policy_list = []
                reward_list = []

        
        if not done:
            
            eval_num_steps_list.append(cnt_steps)

            state = np.expand_dims(state, axis=0)
            _, value = policyValueNetwork(state)
            value = value.numpy()[0][0]
            reward_list.append(value)

            for idx, (state, policy) in enumerate(zip(state_list, policy_list)):
                value = np.dot(reward_list[idx:], discount_factors[:cnt_steps-idx+1])
                value = np.array([value])
                self.add_sample(state, policy, value)

        print(eval_num_steps_list)
        if len(eval_num_steps_list) == 0:
            return max_steps
        else:
            if len(eval_num_steps_list) > 1:
                eval_num_steps_list = eval_num_steps_list[:-1]
                
            return np.average(eval_num_steps_list)
 
    
    def dataset_generator(self):

        for state, policy, value in zip(self.state_list, self.policy_list, self.value_list):
            yield state, policy, value
            

SHUFFLE_WINDOW_SIZE = 1000
BATCH_SIZE = 32
def prepare_data(dataset):
    dataset = dataset.shuffle(SHUFFLE_WINDOW_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
