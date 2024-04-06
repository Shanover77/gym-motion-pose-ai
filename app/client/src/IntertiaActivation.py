import tensorflow as tf

class InertiaActivation(tf.keras.layers.Layer):
    """
    Custom activation function that applies inertia to the input tensor.

    Args:
        threshold (float): The threshold value for significant change.
        inertia_factor (float): The factor to scale the input tensor for inertia.
        decay_factor (float): The factor to scale the after-effects of significant change.

    Returns:
        Tensor: The output tensor after applying the inertia activation function.
    """

    def __init__(self, threshold=0.2, inertia_factor=0.1, decay_factor=0.2, **kwargs):
        super(InertiaActivation, self).__init__(**kwargs)
        self.threshold = threshold
        self.inertia_factor = inertia_factor
        self.decay_factor = decay_factor

    def build(self, input_shape):
        # No trainable parameters for this custom activation function
        super(InertiaActivation, self).build(input_shape)

    def call(self, x):
        inertia_term = tf.keras.activations.sigmoid(self.inertia_factor * x)
        significant_change = tf.keras.activations.relu(x - self.threshold)
        after_effects = tf.keras.activations.exponential(self.decay_factor * (x - 0.1))
        return x + inertia_term * significant_change * after_effects

    def get_config(self):
        config = {
            'threshold': self.threshold,
            'inertia_factor': self.inertia_factor,
            'decay_factor': self.decay_factor,
        }
        base_config = super(InertiaActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Example of how to use the InertiaActivation layer in a model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu'),
#     InertiaActivation(),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# # Compile and train the model as needed
