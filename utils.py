from collections import Counter
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from pathlib import Path
import json
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

def import_dataset(dataset, filename):
    print(f"importing dataset {filename}")
    with open("dataset_config.json", "r") as datasets:
        dataset = json.load(datasets)[dataset]

    csv_path = Path(dataset["path"])/filename
    sep = dataset["files"][filename].get("sep", ",")

    df = pd.read_csv(csv_path, sep=sep)

    if "columns" in dataset["files"][filename]:
        df.columns = dataset["files"][filename]["columns"]

    return df


def yoochoose_sparse_interaction_matrix():
    clicks_df = import_dataset("YOOCHOOSE", "yoochoose-clicks.dat")
    buys_df = import_dataset("YOOCHOOSE", "yoochoose-buys.dat")

    print("assigning contiguous ids")
    sess_id_to_idx = {k: v for v, k in enumerate(set(clicks_df["sessionId"])|set(buys_df["sessionId"]))}
    item_id_to_idx = {k: v for v, k in enumerate(set(clicks_df["itemId"])|set(buys_df["itemId"]))}

    print("assigned embedding indexes to session and item ids")

    pos_session_indices = np.array([sess_id_to_idx[x] for x in buys_df["sessionId"]])
    pos_item_indices = np.array([item_id_to_idx[x] for x in buys_df["itemId"]])

    neg_session_indices_raw = [sess_id_to_idx[x] for x in clicks_df["sessionId"]]
    neg_item_indices_raw = [item_id_to_idx[x] for x in clicks_df["itemId"]]

    # filter out pos values from neg data
    print("filtering out pos values from neg values")
    pos_data = set(zip(pos_session_indices, pos_item_indices))

    neg_session_indices = []
    neg_item_indices = []

    for sess, item in tqdm(zip(neg_session_indices_raw, neg_item_indices_raw)):
        if (sess, item) not in pos_data:
            neg_session_indices.append(sess)
            neg_item_indices.append(item)

    neg_session_indices, neg_item_indices = np.array(neg_session_indices), np.array(neg_item_indices)

    session_indices = np.concatenate((pos_session_indices, neg_session_indices))
    item_indices = np.concatenate((pos_item_indices, neg_item_indices))
    ys = np.concatenate(
        (np.ones(len(pos_session_indices), dtype=np.int8),
         np.zeros(len(neg_session_indices), dtype=np.int8))
    )

    print("shuffling")
    dataset = np.vstack((session_indices, item_indices, ys))
    indices = np.arange(len(session_indices))
    print("reordering")
    dataset = dataset[:, indices]
    session_indices, item_indices, ys = dataset

    Xs = np.vstack((session_indices, item_indices)).T

    return Xs, ys, (sess_id_to_idx, item_id_to_idx), (clicks_df, buys_df)

def sequential_groupby(dataframe, idx_column, other_columns):
    """dataframe groupby assuming column values are sequential. returns rows in order entered.
    
    Arguments:
        dataframe
        idx_column string -- dataframe column id to group by. must be sequential
        other_columns list(string) -- columns to save in output.
    
    Returns:
        list -- grouped rows
    """
    dataset = {}
    cur_id = dataframe[idx_column][0]
    cur_group = []
    for row_id, row in zip(dataframe[idx_column], zip(*[dataframe[column].values.tolist() for column in other_columns])):
        if cur_id == row_id:
            cur_group.append(row)
        else:
            dataset[cur_id] = cur_group

            cur_group = [row]
            cur_id = row_id

    dataset[cur_id] = cur_group

    return dataset

def contiguize_column(dataframe, column, keymap=None):
    """turns the keys in a dataframe column into indexes
    
    Arguments:
        dataframe dataframe -- dataframe
        column string -- column name
        keymap dict -- column key map generated from previous run (default: {None})
    """
    if keymap == None:
        keymap = {}
    
    noncontiguous_keys = dataframe[column].unique()
    
    cur_index = max(keymap.values(), default=-1)+1
    for key in noncontiguous_keys:
        if key not in keymap:
            keymap[key] = cur_index
            cur_index+= 1
    
    f = lambda key: keymap[key]
    dataframe[column] = dataframe[column].apply(f)

    return dataframe, keymap

def load_yoochoose_dataset(reinitialize=False):
    """loads yoochoose dataset. 
    dataset consists of list of sessions. 
    each session is a list of (itemId, weight) pairs. weight=2 means it was a buy. weight=1 means it was a click
    
    Keyword Arguments:
        reinitialize bool -- load dataset over and write over saved values (default: {False})
    
    Returns:
        dataset, column names, keymap -- keymap is a map from contiguous item keys to raw item keys in the dataset
    """
    pickle_save_path = Path("yoochoose.pickle")

    if not reinitialize and pickle_save_path.exists():
        print("found yoochoose cache")
        with open(pickle_save_path, "rb") as file:
            padded_output, output, column_names, keymap = pickle.load(file)
    else:
        print("initializing yoochoose dataset...")

        column_names = ["timestamp", "itemId", "weight"]

        print("loading click dataset...")
        clicks_df = import_dataset("YOOCHOOSE", "yoochoose-clicks.dat")
        clicks_df.insert(len(clicks_df.columns), column="weight", value=2)
        clicks_df, keymap = contiguize_column(clicks_df, "itemId", {})
        grouped_clicks = sequential_groupby(clicks_df, "sessionId", column_names)
        print(f"found {len(clicks_df)} click sessions")

        print("loading buy dataset...")
        buys_df = import_dataset("YOOCHOOSE", "yoochoose-buys.dat")
        buys_df.insert(len(buys_df.columns), column="weight", value=1)
        buys_df, keymap = contiguize_column(buys_df, "itemId", keymap)
        grouped_buys = sequential_groupby(buys_df, "sessionId", column_names)
        print(f"found {len(buys_df)} buy sessions")

        print(f"found {len(keymap)} unique keys")
        keymap = {v: k for (k, v) in keymap.items()}

        output = []
        print("merging datasets...")
        for sessionId, values in grouped_clicks.items():
            if sessionId in grouped_buys:
                output.append(values + grouped_buys[sessionId])
        for sessionId, values in grouped_buys.items():
            if sessionId not in grouped_clicks:
                output.append(values)

        print("sorting session by timestamp, removing timestamps and converting to np")
        assert(column_names[0] == "timestamp")
        column_names = column_names[1:]
        for i in range(len(output)):
            output[i] = np.array([i[1:] for i in sorted(output[i])])

        print("shuffling dataset...")
        random.shuffle(output)

        print("getting max session length")
        max_sess_len = max([len(i) for i in output])
        max_sess_len = 8

        print(f"padding dataset so all sessions length {max_sess_len}")
        padded_output = np.zeros((len(output), max_sess_len, 2), dtype=np.uint32)
        for i in range(len(output)):
            padded_output[i][:min(max_sess_len, len(output[i]))] = output[i][:min(max_sess_len, len(output[i]))]

        print("saving output...")
        with open(pickle_save_path, "wb+") as file:
            pickle.dump((padded_output, output, column_names, keymap), file)

    return padded_output, output, column_names, keymap


def train_val_test_split(*Xs, train_perc=.64, val_perc=.16):
    num_elems = len(Xs[0])

    train_cutoff = int(num_elems*train_perc)
    val_cutoff = int(num_elems*(train_perc+val_perc))

    split_Xs = [(X[:train_cutoff], X[train_cutoff:val_cutoff], X[val_cutoff:]) for X in Xs]

    train = [X[0] for X in split_Xs]
    val = [X[1] for X in split_Xs]
    test = [X[2] for X in split_Xs]

    return train, val, test


def batchify(*args, batch_size=1000, arg_len=None):
    if batch_size == -1:
        yield args

    num_elems = len(args[0]) if arg_len == None else arg_len
    for i in range(0, num_elems, batch_size):
        yield [arg[i: i+batch_size] for arg in args]



# def dataframe_to_interaction_matrix(df, row_cname, col_cname):
#     sess_id_to_idx = {k: v for v, k in enumerate(set(df[row_cname]))}
#     item_id_to_idx = {k: v for v, k in enumerate(set(df[col_cname]))}

#     matrix_shape = (len(sess_id_to_idx), len(item_id_to_idx))

#     row_indices = [sess_id_to_idx[row_id] for row_id in df[row_cname]]
#     col_indices = [item_id_to_idx[col_id] for col_id in df[col_cname]]
#     elements = np.ones(len(col_indices), dtype=np.int8)

#     interaction_matrix = csr_matrix((elements, (row_indices, col_indices)))
    
#     print("count of session-item pairs that exist several times:")
#     print(Counter(interaction_matrix.data))

#     # sometimes the same item is viewed multiple times during a session
#     interaction_matrix = interaction_matrix > 0

#     return interaction_matrix, sess_id_to_idx, item_id_to_idx
