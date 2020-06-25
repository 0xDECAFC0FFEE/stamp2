import tensorflow as tf
from tensorflow.keras.layers import Dense
class LocalAttention(tf.keras.layers.Layer):
    def __init__(self, max_session_len, embedding_size):
        super(LocalAttention, self).__init__()

        print("max_session_len", max_session_len, "embedding_size", embedding_size)
        self.max_session_len = max_session_len
        self.embedding_size = embedding_size

        self.tanh_layer = Dense(
            self.embedding_size,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            name="tanh_layer")

        u_shape = [self.embedding_size]
        self.u = self.add_weight("importance", shape=u_shape)

    def call(self, values, mask):
        batch_size, cur_session_len, item_shape = values.shape[0],values.shape[1],values.shape[2:]
        item_dims = len(item_shape)

        tanh_layer = self.tanh_layer(values)

        similarity_vector = tf.tensordot(tanh_layer, self.u, axes=([2], [0]))

        similarity_vector = similarity_vector+mask

        weights = tf.nn.softmax(similarity_vector, axis=1)
        
        values_transpose_axes = [i+2 for i in range(item_dims)]+[0, 1]
        inv_values_transpose_axes = [item_dims, item_dims+1]+[i for i in range(item_dims)]
        weighted_inputs = tf.transpose(
            tf.transpose(values, perm=values_transpose_axes)*weights,
            perm=inv_values_transpose_axes
        )
        output = tf.math.reduce_sum(weighted_inputs, axis=1)
        return output