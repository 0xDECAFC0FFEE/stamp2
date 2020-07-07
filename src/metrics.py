import tensorflow as tf
import numpy as np

def sequential_mrr(preds, ys, max_rank=20):
    """oracle mrr computation
    """
    mrr = 0
    for y, pred in zip(ys, preds):
        rank = pred.argsort()
        rank[rank.copy()] = np.arange(len(rank), 0, -1)
        rank = rank[y]
        if rank <= max_rank:
            mrr += 1/rank
    return mrr, len(ys)

def numpy_mrr(preds, ys, max_rank=20):
    """testing MRR computation
    """
    n, n_classes = preds.shape

    sort_indices = np.argsort(preds)
    ranks = np.zeros(preds.shape)
    rank_values = np.arange(len(preds[0]), 0, -1)
    np.put_along_axis(ranks, sort_indices, rank_values, axis=1)
    ranks = ranks[np.arange(len(ys)), ys]
    rr = (1/ranks) * (ranks <= max_rank)
    return np.sum(rr), n

@tf.function
def vectorized_mrr(preds, ys, max_rank):
    """computes MRR super speedy

    Args:
        preds (2d np.array): model prediction probabilities
        ys (1d array): true indices
        max_rank (int): maximum rank

    Returns:
        float: mean reciprocal rank
    """
    n, n_classes = preds.shape

    # y_index_2d = tf.stack((tf.range(n, dtype=tf.int32), ys), axis=1)
    # y_probs = tf.expand_dims(tf.gather_nd(preds, y_index_2d), 1)
    # ranks = tf.math.reduce_sum(tf.cast(preds > y_probs, tf.int32), axis=1)+1

    argsort_indices = tf.argsort(preds)
    axis_index = tf.repeat(tf.range(n)[:, None], n_classes, axis=1)
    indices_2d = tf.reshape(tf.stack([axis_index, argsort_indices], 2), [-1, 2])

    scatter_updates = tf.tile(tf.range(n_classes, 0, -1), (n,))
    ranks = tf.scatter_nd(indices_2d, scatter_updates, preds.shape)

    ranks = tf.map_fn(lambda x: x[1:][x[0]], tf.concat([ys[:, None], ranks], 1))

    rr = tf.cast(1/ranks, tf.float32) * tf.cast(ranks <= max_rank, tf.float32)
    return tf.math.reduce_sum(rr), n

class MrrMetric(tf.Module):
    """
    tf.keras.metrics.acc style metric function that computes the mean reverse rank
    """
    def __init__(self, k):
        self.k = tf.Variable(k)
        self.sum = tf.Variable(0.0, dtype=tf.float32)
        self.n = tf.Variable(0)

    @tf.function
    def reset_states(self):
        self.sum.assign(0.0)
        self.n.assign(0)

    @tf.function
    def update_state(self, ys, preds):
        update_sum, update_n = vectorized_mrr(preds, ys, self.k)
        self.sum.assign(update_sum + self.sum)
        self.n.assign(update_n + self.n)

    @tf.function
    def result(self):
        return self.sum/tf.cast(self.n, tf.float32)
