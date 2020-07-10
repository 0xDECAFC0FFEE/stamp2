from pathlib import Path
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from src import utils
from src.features.load_dataset import import_stamp_yoochoose
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import os
import shutil

from tensorflow import keras
from tqdm.notebook import tqdm
from collections import defaultdict
import functools
import copy
from itertools import chain
from src.models import TestModel
from src.models import stmp
from src.metrics import MrrMetric
import json

def experiment(model, config, train_data, val_data):

    X_train, y_train = train_data
    X_val, y_val = val_data
    num_x_train = len(X_train)

    train_metrics = {
        "acc": tf.keras.metrics.SparseCategoricalAccuracy(),
        "top10 acc": tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10),
    }
    val_metrics = {
        "top10 acc": tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10),
    }
    test_metrics = {
        "acc": tf.keras.metrics.SparseCategoricalAccuracy(),
        "top5 acc": tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5),
        "top10 acc": tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10),
        "top20 acc": tf.keras.metrics.SparseTopKCategoricalAccuracy(k=20),
        "mrr5": MrrMetric(k=5),
        "mrr10": MrrMetric(k=10),
        "mrr20": MrrMetric(k=20)
    }

    train_metrics_rec = defaultdict(lambda: [])
    val_metrics_rec = defaultdict(lambda: [])
    test_metrics_rec = defaultdict(lambda: [])

    print(f"starting experiment \"{config['expr_name']}\"")
    expr_dir = Path("logs")/config['expr_name']
    train_log_dir = expr_dir/"train"
    val_log_dir = expr_dir/"val"
    test_log_dir = expr_dir/"test"

    os.makedirs(expr_dir)
    for tensorboard_dir in [train_log_dir, val_log_dir, test_log_dir]:
        os.makedirs(tensorboard_dir, exist_ok=True)

    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
    val_summary_writer = tf.summary.create_file_writer(str(val_log_dir))
    if config["save_model"]:
        model_saver = utils.TopModelSaver(Path("models")/config['expr_name'], config)
    else:
        model_saver = utils.TopModelSaver(Path("models")/"last_run_model", config)

    print(f"batch_size: {config['batch_size']}")
    for epoch in tqdm(list(range(config["epochs"]))):
        print(f"epoch {epoch}")

        # Reset the metrics at the start of the next epoch
        for metric in utils.flatten([train_metrics.values(), val_metrics.values(), test_metrics.values()]):
            metric.reset_states()

        # train model
        for i, (X, y) in enumerate(utils.batchify(X_train, y_train, shuffle=True, batch_size=config["batch_size"])):
            X = [sess[::-1] for sess in X] # reversing sessions so most recent item is last
            X, mask = utils.mask_length(X, maskoff_vals=config["maskoff"], maskon_vals=config["maskon"], justify="right")
            preds, loss = model.train_step(tf.constant(X), tf.constant(mask), tf.constant(y), list(train_metrics.values())) 

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss.numpy(), step=epoch*(int(num_x_train/config["batch_size"])+1)+i)

        # test against validation data
        for X, y in utils.batchify(X_val, y_val, shuffle=True, batch_size=config["batch_size"]):
            X = [sess[::-1] for sess in X]
            X, mask = utils.mask_length(X, maskoff_vals=config["maskoff"], maskon_vals=config["maskon"], justify="right")
            model.test_step(tf.constant(X), tf.constant(mask), tf.constant(y), list(val_metrics.values()))
        # with val_summary_writer.as_default():
            # tf.summary.histogram("preds", preds, step=epoch) # intentionally only getting the last preds cause it takes a while

        with train_summary_writer.as_default():
            for label, metric in train_metrics.items():
                train_metrics_rec[label].append(metric.result())
                # print(f"{label} train: {metric.result()}")
                tf.summary.scalar(label, metric.result(), step=epoch)
        with val_summary_writer.as_default():
            for label, metric in val_metrics.items():
                val_metrics_rec[label].append(metric.result())
                # print(f"{label} val: {metric.result()}")
                tf.summary.scalar(label, metric.result(), step=epoch)

        model_saver.save_best(model, val_metrics["top10 acc"].result())

        # cutoff learning if the score is too low at the early cutoff
        if epoch >= config["early_cutoff_epoch"]:
            if val_metrics["top10 acc"].result() < config["early_cutoff_score"]:
                break

    return model, model_saver
