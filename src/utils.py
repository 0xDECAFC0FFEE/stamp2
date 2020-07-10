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
import datetime
import shutil

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
    nan_val = tf.constant(nan_val, dtype=tf.float32)

    sess_lengths = tf.math.reduce_sum(bool_mask, axis=axis)
    axis_sum = tf.math.reduce_sum(values * bool_mask, axis=axis)
    mean = axis_sum/sess_lengths
    mean = mean * tf.cast(sess_lengths != 0, dtype=tf.float32)
    mean = tf.where(tf.math.is_nan(mean), nan_val, mean)

    return mean

class TopModelSaver():
    def __init__(self, location, config):
        self.prev_best = -np.inf
        
        self.root_folder = location
        if self.root_folder.exists():
            shutil.rmtree(self.root_folder)
        self.model_weights_path = self.root_folder/"model_weights.h5py"
        self.config_path = self.root_folder/"config.json"
        self.source_code_path = self.root_folder/Path(config["file_loc"]).name
        self.saved_config = config

    def reset(self):
        self.prev_best = -np.inf

    def save_best(self, model, score):
        """
        saves best model according to score
        """

        if score > self.prev_best:
            print(f"new best score: {score}; saving weights @ {self.root_folder}")
            if not self.root_folder.exists():
                os.makedirs(self.root_folder)
                with open(self.config_path, "w+") as fp_handle:
                    json.dump(self.saved_config, fp_handle)
                shutil.copyfile(self.saved_config["file_loc"], self.source_code_path)

            model.save_weights(str(self.model_weights_path), save_format="h5")
            self.prev_best = score
        else:
            print(f"cur score {score}. best score remains {self.prev_best}; not saving weights")

def flatten(iterable, max_depth=np.inf):
    """recursively flattens all iterable objects in iterable.

    Args:
        iterable (iterable or numpy array): iterable to flatten
        max_depth (int >= 0, optional): maximum number of objects to iterate into. Defaults to infinity.

    >>> flatten(["01", [2, 3], [[4]], 5, {6:6}.keys(), np.array([7, 8])])
    ['0', '1', 2, 3, 4, 5, 6, 7, 8]

    >>> utils.flatten(["asdf"], max_depth=0)
    ['asdf']

    >>> utils.flatten(["asdf"], max_depth=1)
    ['a', 's', 'd', 'f']
    """
    def recursive_step(iterable, max_depth):
        if max_depth == -1:
            yield iterable
        elif type(iterable) == str:
            for item in iterable:
                yield item
        elif type(iterable) == np.ndarray:
            for array_index in iterable.flatten():
                for item in recursive_step(array_index, max_depth=max_depth-1):
                    yield item
        else:
            try:
                iterator = iter(iterable)
                for sublist in iterator:
                    for item in recursive_step(sublist, max_depth=max_depth-1):
                        yield item
            except (AttributeError, TypeError):
                yield iterable

    assert(max_depth >= 0)
    return recursive_step(iterable, max_depth)

def get_experiment_name(desc=None):
    time = datetime.datetime.now().strftime("%Y-%m-%d | %H:%M:%S |")
    while True:
        if desc == None:
            desc = input("experiment name: ").strip()
        
        if "/" in desc:
            print("please don't use /")
            desc = None
        else:
            return f"expr {time} \"{desc}\""