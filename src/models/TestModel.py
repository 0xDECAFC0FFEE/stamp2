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

def default_config(n_items, max_session_len, **kwargs):
    config = {
        "n_items": n_items,
        "max_session_len": max_session_len,
        "embedding_size": 256,
        "gru_size": 128,
        "dense2_size": 512,
        "dense3_size": 256,
        "softmax_classes": 128,
        "temp": .01,
        "learning_rate": 0.003,
        "maskon": 0,
        "maskoff": -np.inf,
        "batch_size": 1024,
        "epochs": 260,
    }
    config.update(kwargs)
    return config

class model(Model):
    def __init__(self, config):
        super().__init__()
        with open("logs/log.txt", "w+") as handle:
            pass

        self.n_items = config["n_items"]
        embedding_mtx_shape = [config["n_items"], config["embedding_size"]]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])

        self.word_embedding_mtx = self.add_weight(
            initializer=tf.random_uniform_initializer(minval=-1, maxval=1), 
            shape=embedding_mtx_shape,
        )

        self.rnn = Bidirectional(GRU(config["gru_size"], return_sequences=True), merge_mode="concat")
        self.attention1 = LocalAttention(config["max_session_len"], config["gru_size"])
        self.dense2 = Dense(
            config["dense2_size"], 
            activation="linear", 
            kernel_initializer='he_normal'
        )
        self.dense2_act=tf.keras.layers.LeakyReLU()


        self.clustering = Dense(
            config["softmax_classes"],
            activation="linear", 
            kernel_initializer='GlorotNormal'
        )
        self.clustering_act=GrumbelSoftmaxActivation(config["temp"])
        self.clustering_map = Dense(
            config["embedding_size"], 
            activation="linear", 
            kernel_initializer='he_normal'
        )
        self.clustering_map_act=tf.keras.layers.LeakyReLU()

        self.dense3 = Dense(
            config["dense3_size"], 
            activation="linear", 
            kernel_initializer='he_normal'
        )

        self.dense3_act=tf.keras.layers.LeakyReLU()
        self.logits = Dense(
            config["n_items"],
            activation="linear",
            kernel_initializer='GlorotNormal'
        )

    @tf.function
    def call(self, x, mask, training=False, experimental_relax_shapes=True):
        self.word_embedding_mtx[0].assign(tf.zeros(self.word_embedding_mtx[0].shape))
        word_embeddings = tf.nn.embedding_lookup(self.word_embedding_mtx, x)
        
        rnn_output = self.rnn(word_embeddings)
        attention_output = self.attention1(rnn_output, mask)

        if training:
            dense2 = Dropout(.5)(self.dense2_act(self.dense2(attention_output)))
            clusters = self.clustering_act(self.clustering(dense2))
            clusterout = self.clustering_map_act(self.clustering_map(clusters))
            # cluster_att = tf.concat((word_embeddings, clusterout), axis=2)
            cluster_att = tf.concat((dense2, clusterout), axis=1)

            dense3 = Dropout(.5)(self.dense3_act(self.dense3(cluster_att)))

            logits = self.logits(dense3)

        else:
            dense2 = self.dense2_act(self.dense2(attention_output))
            clusters = self.clustering_act(self.clustering(dense2))
            clusterout = self.clustering_map_act(self.clustering_map(clusters))
            cluster_att = tf.concat((dense2, clusterout), axis=1)

            dense3 = self.dense3_act(self.dense3(cluster_att))

            logits = self.logits(dense3)
            
            noise = tf.random.uniform(logits.shape, maxval=10e-6)
            logits = logits + noise # adding the randomness cause topk categorical acc shitty
        
        softmax = tf.nn.softmax(logits)
        return logits, softmax

    @tf.function
    def get_loss(self, y_true, logits):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, logits)

        l1_reg = sum([tf.reduce_sum(tf.math.abs(tf.reshape(weight, [-1]))) for weight in self.trainable_variables])        
        num_trainable_variables = np.sum([np.prod(var.shape) for var in self.trainable_variables])
        l1_reg /= num_trainable_variables
        l1_reg *= 10
        loss = ce_loss+l1_reg
        return loss

    # @tf.function
    def train_step(self, sessions, mask, labels, scores):
        with tf.GradientTape() as tape:
            logits, preds = self.call(sessions, mask, training=True)
            loss = self.get_loss(y_true=labels, logits=logits)
        gradients = tape.gradient(loss, self.trainable_variables)
        with open("logs/log.txt", "a+") as handle:
            # print("trainable vars", [i.shape for i in self.trainable_variables], file=handle)
            print("grads", [i.numpy() for i in gradients], file=handle)
        # gradients = [tf.clip_by_value(grad, -1., 1.) for grad in gradients] # clip grads to stop nan problem
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
