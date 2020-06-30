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


class STMP_model(Model):
    def __init__(self, config):
        super().__init__()
        self.n_items = config["n_items"]
        embedding_size = config["embedding_size"]
        embedding_mtx_shape = [self.n_items, embedding_size]

        self.item_embedding_mtx = self.add_weight(
            initializer=tf.random_uniform_initializer(minval=-1, maxval=1), 
            shape=embedding_mtx_shape,
        )

        self.hs = Dense(
            embedding_size, 
            activation="tanh"
        )

        self.ht = Dense(
            embedding_size, 
            activation="tanh"
        )

    def call(self, X, mask, training=False):
        X_embeddings = tf.nn.embedding_lookup(self.item_embedding_mtx, X_s)
        X_s = X_embeddings[:-1]
        x_s = tf.reduce_mean(X_s, axis=1)
        x_t = X_embeddings[-1]
        self.hs(xs)
        self.ht(xt)