import gym
import numpy as np
import tqdm
import datetime

from MCTS import *
from PolicyValueNetwork import *
from ReplayMemory import *
from Logging import *

# openai gym causes a warning - disable it
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

num_train_steps = 100

max_steps = 350
gamma = 0.9
discount_factors = np.array([gamma**i for i in range(max_steps + 1)])

def fill(policyValueNetwork, replayMemory, num_trajectories):
    env = gym.make("CartPole-v1") 
    state, _ = env.reset()

    eval_num_steps_list = []

    state_list = []
    policy_list = []
    reward_list = []

    done = False
  
    cnt_steps = 0

    for i in tqdm.tqdm(range(num_trajectories),position=0, leave=True):
        mcts = MCTS(env, state, policyValueNetwork)

        policy = mcts.run(200)

        state_list.append(state)
        policy_list.append(policy)
        
        # randomly choose action (not the best one!)
        action = np.random.choice(2, p=policy)
        state, reward, done, _, _ = env.step(action)
        reward_list.append(reward)

        cnt_steps += 1
             
        if done or cnt_steps == max_steps:
        
            eval_num_steps_list.append(cnt_steps)
                
            state, _ = env.reset()

            for idx, (state, policy) in enumerate(zip(state_list, policy_list)):
                reward_list_len = len(reward_list[idx:])
                value = np.dot(reward_list[idx:], discount_factors[:reward_list_len])
                value = np.array([value])
                
                replayMemory.add_sample(state, policy, value)
                 
            cnt_steps = 0
            state_list = []
            policy_list = []
            reward_list = []


    if not done:
            
        state = np.expand_dims(state, axis=0)
        _, value = policyValueNetwork(state)
        value = value.numpy()[0][0]
        reward_list.append(value)

        for idx, (state, policy) in enumerate(zip(state_list, policy_list)):
            reward_list_len = len(reward_list[idx:])
            value = np.dot(reward_list[idx:], discount_factors[:reward_list_len])
            value = np.array([value])
            replayMemory.add_sample(state, policy, value)

    print(eval_num_steps_list)
    if len(eval_num_steps_list) == 0:
        return max_steps
    else:     
        return np.average(eval_num_steps_list)


def main():

    #
    # Network
    #
    policyValueNetwork = PolicyValueNetwork()
    policyValueNetwork.build(input_shape=(1, 4))
    policyValueNetwork.summary()

    #
    # Logging
    #
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    #
    # Replay memory: put some init trajectories in
    #
    replayMemory = ReplayMemory(2000)
    num_steps = fill(policyValueNetwork, replayMemory, 1000)
  
    for train_step in range(num_train_steps):

        # was initial filled
        if train_step != 0:
            num_steps = fill(policyValueNetwork, replayMemory, 500)

        #
        # Train network
        #

        # Replay memory -> dataset
        dataset = tf.data.Dataset.from_generator(
                        replayMemory.dataset_generator,
                        output_signature=(
                                tf.TensorSpec(shape=(4,), dtype=np.float32),
                                tf.TensorSpec(shape=(2,), dtype=tf.float32),
                                tf.TensorSpec(shape=(1,), dtype=np.float32)
                            )
                    )
        
        dataset = dataset.apply(prepare_data)

        # Train steps
        for num_epoch in range(10):
            for state, target_policy, target_value in tqdm.tqdm(dataset,position=0, leave=True):
                policyValueNetwork.train_step(state, target_policy, target_value)
        

        log(train_summary_writer, policyValueNetwork, num_steps, train_step)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
