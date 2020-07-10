import tensorflow as tf
import numpy as np
import sys
from src.models.GrumbelSoftmaxActivation import GrumbelSoftmaxActivation
from src.models.LocalAttention import LocalAttention
from src.models.trilinear_product import trilinear_product
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Flatten, Dense, Embedding, RNN, GRU, Bidirectional, Layer, Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from src import utils
import datetime
from pathlib import Path

def default_config(n_items, **kwargs):
    config = {
        "n_items": n_items,
        "embedding_size": 100,
        "learning_rate": 0.005,
        "maskon": 1,
        "maskoff": 0,
        "batch_size": 512,
        "epochs": 30,
        "embed_std": 0.002,
        "fc_std": 0.05,
        "save_model": True,
        "decay_lr": False,
        "decay_steps": 28, # steps between step
        "decay_rate": .95,
        "num_dense_layers": 1,
        "early_cutoff_score": .12,
        "early_cutoff_epoch": 50,
        "file_loc": str(Path(__file__).resolve()),
    }
    config.update(kwargs)

    if "expr_name" in config:
        config["expr_name"] = utils.get_experiment_name(config["expr_name"])
    else:
        config["expr_name"] = utils.get_experiment_name()

    return config

class model(Model):
    def __init__(self, config):
        super().__init__()        
        with open("logs/log.txt", "w+") as handle:
            pass
        self.n_items = config["n_items"]
        embedding_mtx_shape = [self.n_items, config["embedding_size"]]

        if config["decay_lr"]:
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=tf.constant(config["learning_rate"]), 
                decay_steps=tf.constant(config["decay_steps"]),
                decay_rate=tf.constant(config["decay_rate"]),
                staircase=True
            )
        else:
            learning_rate = config["learning_rate"]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.item_embs = self.add_weight(
            initializer=tf.random_normal_initializer(mean=0.0, stddev=config["embed_std"]), 
#             initializer=tf.random_uniform_initializer(minval=-1, maxval=1), 
            shape=embedding_mtx_shape,
            dtype=tf.float32
        )

        self.hs = [] # allowing dense layer multiple hidden layers
        for _ in range(config["num_dense_layers"]):
            self.hs.append(Dense(
                config["embedding_size"], 
                activation="tanh",
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=config["fc_std"]),
                bias_initializer=tf.zeros_initializer(),
                dtype=tf.float32
            ))

        self.ht = [] # allowing dense layer multiple hidden layers
        for _ in range(config["num_dense_layers"]):
            self.ht.append(Dense(
                config["embedding_size"], 
                activation="tanh",
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=config["fc_std"]),
                bias_initializer=tf.zeros_initializer(),
                dtype=tf.float32
            ))

    @tf.function
    def call(self, sessions, mask, training=False):
        self.item_embs[0].assign(tf.zeros(self.item_embs[0].shape))

        X_embeddings = tf.nn.embedding_lookup(self.item_embs, sessions)
        # print("mask", mask)
        # print("X_embeddings", X_embeddings)

        X_s = X_embeddings[:, :-1]
        # print("X_s", X_s)
        # print("mask, reversed, last value removed", mask[:, :-1, None])
        ms = utils.masked_mean(X_s, mask[:, :-1, None])
        # print("ms", ms)
        mt = X_embeddings[:, -1]
        # print("mt", mt)

        hs_val = ms
        for ht_layer in self.hs: # giving dense layer several hidden layers
            hs_val = ht_layer(hs_val)

        ht_val = mt
        for ht_layer in self.ht: # giving dense layer several hidden layers
            ht_val = ht_layer(ht_val)

        prod = trilinear_product(hs_val, ht_val, self.item_embs)
        # print("trilinear_product", prod)
        logits = tf.math.sigmoid(prod)
        # print("logits", logits)

        noise = tf.random.uniform(logits.shape, maxval=10e-8, dtype=tf.float32)
        logits = logits+noise
        # print("logits+noise", logits)

        softmax = tf.nn.softmax(logits, axis=1)
        # print("softmax", softmax)

        return logits, softmax

    @tf.function
    def get_loss(self, y_true, logits):
        # ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, logits)
        ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits))

#         l1_reg = sum([tf.reduce_sum(tf.math.abs(tf.reshape(weight, [-1]))) for weight in self.trainable_variables])        
#         num_trainable_variables = np.sum([np.prod(var.shape) for var in self.trainable_variables])
#         l1_reg /= num_trainable_variables
#         l1_reg *= 10
#         # print(ce_loss, l1_reg)
#         loss = ce_loss+l1_reg

#         # print(ce_loss)
        return ce_loss

    @tf.function
    def train_step(self, sessions, mask, labels, scores):
        with tf.GradientTape() as tape:
            logits, preds = self.call(sessions, mask, training=True)
            loss = self.get_loss(y_true=labels, logits=logits)
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [tf.clip_by_norm(grad, 150.) for grad in gradients] # clip grads to stop nan problem
        # gradients = [tf.clip_by_value(grad, -1., 1.) for grad in gradients] # clip grads to stop nan problem
        # with open("logs/log.txt", "a+") as handle:
            # print("trainable vars", [i.shape for i in self.trainable_variables], file=handle)
            # print("grads", gradients[4], file=handle)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for score in scores:
            score.update_state(labels, tf.cast(preds, tf.float32))

        return preds, loss

    @tf.function
    def test_step(self, sessions, mask, labels, scores):
        logits, preds = self.call(sessions, mask, training=False)
        
        for score in scores:
            score.update_state(labels, tf.cast(preds, tf.float32))
        return preds
