import tensorflow as tf

class PolicyValueNetwork(tf.keras.Model):

    def __init__(self):
     
        super(PolicyValueNetwork, self).__init__()

        self.frontend_layer_list = [
            tf.keras.layers.Dense(25, activation="tanh"),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(20, activation="tanh"),
            tf.keras.layers.BatchNormalization()
        ]

        self.backend_policy_layer_list = [
            tf.keras.layers.Dense(10, activation="tanh"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(2, activation="softmax")
        ]

        self.backend_value_layer_list = [
            tf.keras.layers.Dense(10, activation="tanh"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ]
   

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.cce_loss =  tf.keras.losses.CategoricalCrossentropy()
        self.mse_loss =  tf.keras.losses.MeanSquaredError()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")

        self.metric_policy_loss = tf.keras.metrics.Mean(name="policy_loss")
     
        self.metric_value_loss = tf.keras.metrics.Mean(name="value_loss")

    @tf.function
    def call(self, x, training=False):
        
        for layer in self.frontend_layer_list:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training)
            else:
                x = layer(x)

        policy = x
        for layer in self.backend_policy_layer_list:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                policy = layer(policy, training)
            else:
                policy = layer(policy)

        value = x
        for layer in self.backend_value_layer_list:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                value = layer(value, training)
            else:
                value = layer(value)

        return policy, value 
    

    @tf.function
    def train_step(self, x, target_policy, target_value):
    
        with tf.GradientTape() as tape:
            policy, value = self(x, training=True)

            policy_loss = self.cce_loss(target_policy, policy)
            value_loss = self.mse_loss(target_value, value)

            loss = policy_loss + value_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)

        self.metric_policy_loss.update_state(policy_loss)
        self.metric_value_loss.update_state(value_loss)