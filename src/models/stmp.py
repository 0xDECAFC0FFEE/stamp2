import tensorflow as tf
import numpy as np
import sys
from src.models.GrumbelSoftmaxActivation import GrumbelSoftmaxActivation
from src.models.LocalAttention import LocalAttention
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Flatten, Dense, Embedding, RNN, GRU, Bidirectional, Layer, Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from src import utils

class model(Model):
    def __init__(self, config):
        super().__init__()
        self.n_items = config["n_items"]
        embedding_mtx_shape = [self.n_items, config["embedding_size"]]

        learning_rate = config["learning_rate"]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.item_embs = self.add_weight(
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.002), 
            shape=embedding_mtx_shape,
            dtype=tf.float32
        )

        self.hs = Dense(
            config["embedding_size"], 
            activation="tanh"
        )

        self.ht = Dense(
            config["embedding_size"], 
            activation="tanh"
        )

    @tf.function
    def call(self, sessions, mask, training=False):
        X_embeddings = tf.nn.embedding_lookup(self.item_embs, sessions)
        # tf.print("X_embeddings", X_embeddings)

        X_s = X_embeddings[:, :-1]
        # tf.print("X_s", X_s)
        ms = utils.masked_mean(X_s, mask[:, :-1, None])
        # tf.print("ms", ms)
        mt = X_embeddings[:, -1]
        # tf.print("mt", mt)
        
        hs_val = self.hs(ms)
        # tf.print("hs_val", hs_val)
        ht_val = self.ht(mt)
        # tf.print("ht_val", ht_val)

        prod = utils.trilinear_product(hs_val, ht_val, self.item_embs)
        # tf.print("trilinear_product", prod)
        logits = tf.math.sigmoid(prod)
        # tf.print("logits", logits)

        noise = tf.random.uniform(logits.shape, maxval=10e-6)
        logits = logits+noise
        # tf.print("logits+noise", logits)

        softmax = tf.nn.softmax(logits, axis=1)
        # tf.print("softmax", softmax)

        return logits, softmax

    @tf.function
    def get_loss(self, y_true, logits):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, logits)

        l1_reg = sum([tf.reduce_sum(tf.math.abs(tf.reshape(weight, [-1]))) for weight in self.trainable_variables])        
        num_trainable_variables = np.sum([np.prod(var.shape) for var in self.trainable_variables])
        l1_reg /= num_trainable_variables
        l1_reg *= 10
        tf.print(ce_loss, l1_reg)
        loss = ce_loss+l1_reg

        return loss

    @tf.function
    def train_step(self, sessions, mask, labels, scores):
        with tf.GradientTape() as tape:
            logits, preds = self.call(sessions, mask, training=True)
            loss = self.get_loss(y_true=labels, logits=logits)
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [tf.clip_by_value(grad, -1., 1.) for grad in gradients] # clip grads to stop nan problem
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for score in scores:
            score.update_state(labels, tf.cast(preds, tf.float32))

        return preds

    @tf.function
    def test_step(self, sessions, mask, labels, scores):
        logits, preds = self.call(sessions, mask, training=False)
        
        for score in scores:
            score.update_state(labels, tf.cast(preds, tf.float32))
        return preds
