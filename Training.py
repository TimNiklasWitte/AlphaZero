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

def main():

    policyValueNetwork = PolicyValueNetwork()

    #
    # Logging
    #
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    for i in range(100):
        
        replayMemory = ReplayMemory(300)
        num_steps = replayMemory.fill(policyValueNetwork)

        dataset = tf.data.Dataset.from_generator(
                        replayMemory.dataset_generator,
                        output_signature=(
                                tf.TensorSpec(shape=(4,), dtype=np.float32),
                                tf.TensorSpec(shape=(2,), dtype=tf.float32),
                                tf.TensorSpec(shape=(1,), dtype=np.float32)
                            )
                    )
        
        dataset = dataset.apply(prepare_data)


        for num_epoch in range(5):
            for state, target_policy, target_value in tqdm.tqdm(dataset,position=0, leave=True):
                policyValueNetwork.train_step(state, target_policy, target_value)
        

        log(train_summary_writer, policyValueNetwork, num_steps, i)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
