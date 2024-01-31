import tensorflow as tf
import numpy as np 


def log(train_summary_writer, policyValueNetwork, num_steps, epoch):
    
    #
    # Get losses
    #
    loss = policyValueNetwork.metric_loss.result()
    policy_loss = policyValueNetwork.metric_policy_loss.result()
    value_loss = policyValueNetwork.metric_value_loss.result()

    #
    # Reset
    #
    policyValueNetwork.metric_loss.reset_states()
    policyValueNetwork.metric_policy_loss.reset_states()
    policyValueNetwork.metric_value_loss.reset_states()


    #
    # Write to TensorBoard
    #
    with train_summary_writer.as_default():
        tf.summary.scalar("loss", loss, step=epoch)
        tf.summary.scalar("policy_loss", policy_loss, step=epoch)
        tf.summary.scalar("value_loss", value_loss, step=epoch)

        tf.summary.scalar("num_steps", num_steps, step=epoch)

    #
    # Output
    #
    print(f"       loss: {loss:.5f}")
    print(f"policy_loss: {policy_loss:.5f}")
    print(f" value_loss: {value_loss:.5f}")

    print(f"  num_steps: {num_steps}")

    print()