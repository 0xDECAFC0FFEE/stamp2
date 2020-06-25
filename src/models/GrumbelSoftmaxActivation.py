import tensorflow as tf
from tensorflow_probability import distributions
class GrumbelSoftmaxActivation(tf.keras.layers.Layer):
    def __init__(self, temp):
        super(GrumbelSoftmaxActivation, self).__init__()
        self.temp = temp
        self.gumbel = distributions.Gumbel(0, 1)

    def call(self, values):
        values = tf.nn.softmax(values, axis=1)
        grumbel_sample = self.gumbel.sample(values.shape)
        softmax_input = (tf.math.log(values)+grumbel_sample)/self.temp
        output = tf.nn.softmax(softmax_input, axis=1)
        return output
