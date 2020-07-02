from collections import Counter, defaultdict
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import json
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
import copy
from itertools import chain
from contextlib import contextmanager
import sys
from pathlib import Path
import tensorflow as tf

@contextmanager
def cwd(path):
    """change directory in with block

        with cwd("new/path"):
            pass

    """
    path = Path(path).resolve()
    old_paths = copy.copy(sys.path)
    oldpwd=os.getcwd()
    os.chdir(path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        os.chdir(oldpwd)
        sys.path = old_paths

def mask_length(sessions, maskoff_vals=0, maskon_vals=1, justify="right"):
    """take sessions, turn it into a numpy array and create a mask to ignore portions of sessions that don't matter
    maskoff_vals = -inf for attention
    
    Arguments:
        sessions {list of variable length list of items} -- the sessions to turn into a numpy array. between each session, all dimensions must agree except the second
    
    Keyword Arguments:
        maskoff_vals {int} -- values to use in mask for boolmask session items that don't exist (default: {0})
        maskon_vals {int} -- values to use in mask for boolmask session items that exist (default: {1})
        justify ({string}) -- if "right", short sessions on the rhs, if "left", short sessions on lhs

    Returns:
        sessions_array, mask -- sessions as a numpy array and the mask
    """

    item_shape = sessions[0][0].shape
    num_sessions = len(sessions)
    sess_lengths = [len(sess) for sess in sessions]
    max_sess_len = int((max(sess_lengths)+3)/4)*4 # cutting down the amount of retracing by 4

    if justify == "left":
        mask_bool = np.arange(max_sess_len).reshape(-1, max_sess_len).repeat(num_sessions, axis=0)
        mask_bool = mask_bool < np.array(sess_lengths).reshape(num_sessions, -1)
        mask = np.zeros(mask_bool.shape, np.float32)
        mask[mask_bool == 1] = maskon_vals
        mask[mask_bool == 0] = maskoff_vals

        sessions_array = np.zeros([num_sessions, max_sess_len]+list(item_shape), dtype=sessions[0].dtype)
        for length, i in zip(sess_lengths, range(num_sessions)):
            sessions_array[i, :length] = sessions[i]

    else:
        mask_bool = np.arange(max_sess_len-1, -1, -1).reshape(-1, max_sess_len).repeat(num_sessions, axis=0)
        mask_bool = mask_bool < np.array(sess_lengths).reshape(num_sessions, -1)
        mask = np.zeros(mask_bool.shape, np.float32)
        mask[mask_bool == 1] = maskon_vals
        mask[mask_bool == 0] = maskoff_vals

        sessions_array = np.zeros([num_sessions, max_sess_len]+list(item_shape), dtype=sessions[0].dtype)
        for length, i in zip(sess_lengths, range(num_sessions)):
            sessions_array[i, -length:] = sessions[i]

    return sessions_array, mask

def augment_session_len(sessions):
    """
        augment session by taking each n-1 subsession as a different datapoint.
        leaves the last item along as 
    """

    augmented_sessions = []
    for clicks in sessions:
        for i in range(len(clicks)-1):
            augmented_sessions.append(clicks[:i+1])

    return augmented_sessions

def final_item_class(sessions):
    """
        split session list into first n-1 item list and last item list
    """
    print("using last item in session as class")

    cur_augmented_session_id = 0
    out_sess = []
    y = []
    for session in sessions:
        y.append(session[-1])
        out_sess.append(np.array(session[:-1]))
    return out_sess, y

def train_val_test_split(*Xs, train_perc=.64, val_perc=.16, test_perc=.2):
    """split dataset into train, val and test datasets

    (train_X, train_y), (val_X, val_y), (test_X, test_y) = train_val_test_split(X, y)

    train_data, val_data, _ = train_val_test_split(X, y, train_perc=.8, val_perc=.2, test_perc=0)

    [train_data], [val_data], [test_data] = train_val_test_split(X)

    """
    print(f"train_val_test_split @ {train_perc} train, {val_perc} val, {test_perc} test")
    assert(train_perc+val_perc+test_perc == 1)
    num_elems = len(Xs[0])

    train_cutoff = int(num_elems*train_perc)
    val_cutoff = int(num_elems*(train_perc+val_perc))
    test_cutoff = int(num_elems*(train_perc+val_perc+test_perc))

    split_Xs = [(X[:train_cutoff], X[train_cutoff:val_cutoff], X[val_cutoff:test_cutoff]) for X in Xs]

    train = [X[0] for X in split_Xs]
    val = [X[1] for X in split_Xs]
    test = [X[2] for X in split_Xs]

    return train, val, test

def batchify(*args, batch_size=1000, arg_len=None, shuffle=False):
    if batch_size == -1:
        yield args
    if shuffle:
        args = list(zip(*args))
        random.shuffle(args)
        args = list(zip(*args))
    num_elems = len(args[0]) if arg_len == None else arg_len
    for i in range(0, num_elems, batch_size):
        yield [arg[i: i+batch_size] for arg in args]

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

@tf.function()
def trilinear_product(a, b, c):
    """gets the trilinear product of a, b, and c.
    if a, b, c are [1, d] vectors, <a, b, c> = \Sigma_i a_i * b_i * c_i

    note that the ordering of a, b, and c in the paper is slightly different than
    the implementation here as 1) it disagrees with the actual source code released and
    2) it results in shape errors

    Args:
        a (matrix): [m, d] matrix of average of the first n-1 session item embeddings
        b (matrix): [m, d] matrix of last session item embeddings
        c (matrix): [n, d] matrix of all the item embeddings

    Returns:
        matrix: [m, n] matrix of item scores for each session
    """
    return tf.transpose(c @ tf.transpose(a * b))

@tf.function()
def masked_mean(values, bool_mask, axis=1, nan_val=0.0):
    """mean along axis with masked values

    preds = [[1, 2, 3, 4], [1, 2, 3], []]
    preds = [np.array(sess, dtype=np.float32) for sess in preds]

    padded_preds, mask = utils.mask_length(preds)
    print(utils.masked_mean(padded_preds, mask, axis=1))

    >>> tf.Tensor([2.5 2 0], shape=(3,), dtype=float32)

    Args:
        values (matrix): values to take the mean of
        axis (int): axis to take the mean along (default: {1})
        bool_mask (bool matrix): values to mask off (0=remove, 1=keep)
        nan_val (float): value to replace nans with (default: {0.0})
    """

    sess_lengths = tf.math.reduce_sum(bool_mask, axis=axis)
    axis_sum = tf.math.reduce_sum(values * bool_mask, axis=axis)
    mean = axis_sum/sess_lengths
    mean = mean * tf.cast(sess_lengths != 0, dtype=tf.float32)
    mean = tf.where(tf.math.is_nan(mean), nan_val, mean)

    return mean

def save_best(model, score, location):
    """
    saves best model according to score
    """

    if score > save_best.prev_best:
        print(f"new best score: {score}; saving weights @ {location}")
        model.save_weights(location, save_format="h5")
        save_best.prev_best = score
    else:
        print(f"best score remains {save_best.prev_best}; not saving weights")
save_best.prev_best = -np.inf