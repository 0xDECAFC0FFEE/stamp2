from src.utils import *
from src.features.build_features import save_folder as dataset_folder

def import_stamp_yoochoose(perc):
    assert(perc in [4, 64])
    if perc == 4:
        with open(dataset_folder/"dataset_4.pkl"), "rb" as handle:
            train_data, test_data, idx2item, n_items = pickle.load(handle)
    else:
        with open(dataset_folder/"dataset_64.pkl", "rb") as handle:
            train_data, test_data, idx2item, n_items = pickle.load(handle)

    max_train_session_len = max([len(session) for session in train_data])
    max_test_session_len = max([len(session) for session in test_data])
    max_session_len = max(max_train_session_len, max_test_session_len)
    embedding_sizes = n_items, max_session_len

    train_data = final_item_class(train_data)
    test_data = final_item_class(test_data)

    train_data, val_data, _ = train_val_test_split(*train_data, train_perc=.8, val_perc=.2, test_perc=0)

    return train_data, val_data, test_data, idx2item, embedding_sizes