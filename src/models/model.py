import tensorflow as tf
import sys
from src.models.GrumbelSoftmaxActivation import GrumbelSoftmaxActivation
from src.models.LocalAttention import LocalAttention
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Flatten, Dense, Embedding, RNN, GRU, Bidirectional, Layer, Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb


class TestModel(Model):
    def __init__(self, n_items, max_session_len, embedding_size, gru_size, dense2_size, dense3_size, softmax_classes, temp):
        super(TestModel, self).__init__()
        self.n_items = n_items
        embedding_mtx_shape = [n_items, embedding_size]
        self.word_embedding_mtx = self.add_weight(
            initializer=tf.random_uniform_initializer(minval=-1, maxval=1), 
            shape=embedding_mtx_shape,
        )

        print("embedding_mtx_shape", embedding_mtx_shape)

        self.rnn = Bidirectional(GRU(gru_size, return_sequences=True), merge_mode="concat")
        self.attention1 = LocalAttention(max_session_len, gru_size)
        self.dense2 = Dense(
            dense2_size, 
            activation="linear", 
            kernel_initializer='he_normal'
        )
        self.dense2_act=tf.keras.layers.LeakyReLU()


        self.clustering = Dense(
            softmax_classes,
            activation="linear", 
            kernel_initializer='GlorotNormal'
        )
        self.clustering_act=GrumbelSoftmaxActivation(temp)
        self.clustering_map = Dense(
            embedding_size, 
            activation="linear", 
            kernel_initializer='he_normal'
        )
        self.clustering_map_act=tf.keras.layers.LeakyReLU()




        self.dense3 = Dense(
            dense3_size, 
            activation="linear", 
            kernel_initializer='he_normal'
        )

        self.dense3_act=tf.keras.layers.LeakyReLU()
        self.logits = Dense(
            n_items,
            activation="linear",
            kernel_initializer='GlorotNormal'
        )

    def call(self, x, mask, training=False):
        print("xshape", x.shape.as_list())
        word_embeddings = tf.nn.embedding_lookup(self.word_embedding_mtx, x)
        print("word_embeddings out", word_embeddings.shape.as_list())
        
        
        rnn_output = self.rnn(word_embeddings)
        print("rnn shape", rnn_output.shape.as_list())
        attention_output = self.attention1(rnn_output, mask)
        print("attention shape", attention_output.shape.as_list())

        if training:
            dense2 = Dropout(.5)(self.dense2_act(self.dense2(attention_output)))
            print("dense2 shape", dense2.shape.as_list())
            clusters = self.clustering_act(self.clustering(dense2))
            print("clusters shape", clusters.shape.as_list())
            clusterout = self.clustering_map_act(self.clustering_map(clusters))
            print("cluster shape", clusterout.shape.as_list())
            # cluster_att = tf.concat((word_embeddings, clusterout), axis=2)
            cluster_att = tf.concat((dense2, clusterout), axis=1)
            print("cluster_att shape", cluster_att.shape.as_list())

            dense3 = Dropout(.5)(self.dense3_act(self.dense3(cluster_att)))
            print("dense3 shape", dense3.shape.as_list())

            logits = self.logits(dense3)
            print("logits shape", logits.shape.as_list())

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

    def get_loss(self, y_true, logits):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, logits)

        l1_reg = sum([tf.reduce_sum(tf.math.abs(tf.reshape(weight, [-1]))) for weight in self.trainable_variables])        
        num_trainable_variables = np.sum([np.prod(var.shape) for var in self.trainable_variables])
        l1_reg /= num_trainable_variables
        l1_reg *= 10
        loss = ce_loss+l1_reg
        return loss

    @tf.function
    def train_step(self, sessions, mask, labels, scores):
        with tf.GradientTape() as tape:
            logits, preds = self.call(sessions, mask, training=True)
            loss = self.get_loss(y_true=labels, logits=logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients = [tf.clip_by_value(grad, -1., 1.) for grad in gradients] # clip grads to stop nan problem
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        for score in scores:
            score.update_state(labels, tf.cast(preds, tf.float32))

        return preds

    @tf.function
    def test_step(self, sessions, mask, labels, scores):
        logits, preds = self.call(sessions, mask, training=False)
        for score in scores:
            score.update_state(labels, tf.cast(preds, tf.float32))
        return preds
