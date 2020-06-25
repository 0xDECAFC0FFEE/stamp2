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

@contextmanager
def cwd(path):
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

def mask_length(sessions, maskoff_vals=0, maskon_vals=1):
    """take sessions, turn it into a numpy array and create a mask to ignore portions of sessions that don't matter
    
    Arguments:
        sessions {list of variable length list of items} -- the sessions to turn into a numpy array. between each session, all dimensions must agree except the second
    
    Keyword Arguments:
        maskoff_vals {int} -- values to use in mask for session items that don't exist (default: {0})
        maskon_vals {int} -- values to use in mask for session items that exist (default: {1})

    Returns:
        sessions_array, mask -- sessions as a numpy array and the mask
    """
    item_shape = sessions[0].shape
    num_sessions = len(sessions)
    sess_lengths = [len(sess) for sess in sessions]
    max_sess_len = int((max(sess_lengths)+3)/4)*4 # cutting the amount of retracing in half

    mask_bool = np.arange(max_sess_len).reshape(-1, max_sess_len).repeat(num_sessions, axis=0)
    mask_bool = mask_bool < np.array(sess_lengths).reshape(num_sessions, -1)
    mask = np.zeros(mask_bool.shape, np.float32)
    mask[mask_bool == 1] = maskon_vals
    mask[mask_bool == 0] = maskoff_vals

    sessions_array = np.zeros([num_sessions, max_sess_len]+list(item_shape[1:]), dtype=sessions[0].dtype)
    for length, i in zip(sess_lengths, range(num_sessions)):
        sessions_array[i, :length] = sessions[i]

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
        use last item in session list as the class
    """
    print("using last item in session as class")

    cur_augmented_session_id = 0
    out_sess = []
    y = []
    for session in sessions:
        y.append(session[-1])
        out_sess.append(session[:-1])
    return out_sess, y

def train_val_test_split(*Xs, train_perc=.64, val_perc=.16, test_perc=.2):
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

