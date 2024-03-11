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

num_train_steps = 500

replay_memory_size = 25000
max_steps = 1000
gamma = 0.9
discount_factors = np.array([gamma**i for i in range(max_steps + 1)])

max_return = np.dot(np.ones(shape=(max_steps + 1, )), discount_factors)
max_return = float(max_return)


def fill(policyValueNetwork, replayMemory, max_steps):
    env = gym.make("CartPole-v1") 
    state, _ = env.reset()

    state_list = []
    policy_list = []
    reward_list = []

    done = False
  
    for i in tqdm.tqdm(range(max_steps),position=0, leave=True):
        mcts = MCTS(env, state, policyValueNetwork, max_return)

        policy = mcts.run(400)

        state_list.append(state)
        policy_list.append(policy)
        
        # randomly choose action (not the best one!)
        action = np.random.choice(2, p=policy)
        state, reward, done, _, _ = env.step(action)
        reward_list.append(reward)

        if done:
            
            reward_list = [reward/max_return for reward in reward_list]
            
            # bootstrap
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

            break 
    
    # bootstrap
    if not done:
        
        reward_list = [reward/max_return for reward in reward_list]

        state = np.expand_dims(state, axis=0)
        _, value = policyValueNetwork(state)
        value = value.numpy()[0][0]
        reward_list.append(value)


        for idx, (state, policy) in enumerate(zip(state_list, policy_list)):
            reward_list_len = len(reward_list[idx:])
            value = np.dot(reward_list[idx:], discount_factors[:reward_list_len])
            value = np.array([value])
            replayMemory.add_sample(state, policy, value)

    return i 


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

    print("Step: 0")

    replayMemory = ReplayMemory(replay_memory_size)
    num_steps = fill(policyValueNetwork, replayMemory, max_steps)
    print(f"  num_steps: {num_steps}")
    print()

    # Record step 0 (only num_steps)
    with train_summary_writer.as_default():
        tf.summary.scalar("num_steps", num_steps, step=0)

    for train_step in range(1, num_train_steps):

        print("Step: ", train_step)
        num_steps = fill(policyValueNetwork, replayMemory, max_steps)

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

        for state, target_policy, target_value in tqdm.tqdm(dataset,position=0, leave=True):
            policyValueNetwork.train_step(state, target_policy, target_value)
        

        log(train_summary_writer, policyValueNetwork, num_steps, train_step)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
