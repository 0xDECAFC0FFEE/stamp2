import pickle
from pathlib import Path
import os
import sys
sys.path.append(str(Path().cwd()))
from src.utils import cwd
from itertools import chain

input_click_file = (Path("data")/"raw"/"yoochoose"/"yoochoose-clicks.dat").resolve()
STAMP_folder = (Path("src")/"features"/"STAMP").resolve()
save_folder = (Path("data")/"processed"/"STAMP-yoochoose").resolve()

def dataset_statistics(balazs_train, balazs_test, dataset_64, dataset_4):
    """print dataset statistics to verify it's the same as the stamp paper
    """
    print("=== dataset statistics ===")
    print("post filter, pre sequence splitting dataset statistics:")
    print(f"num train sessions: {balazs_train.SessionId.nunique()}")
    print(f"num train clicks: {len(balazs_train)}")
    print(f"num train items: {balazs_train.ItemId.nunique()}")

    print(f"num test sessions: {balazs_test.SessionId.nunique()}")
    print(f"num test clicks: {len(balazs_test)}")
    print(f"num test items: {balazs_test.ItemId.nunique()}")

    def ds_stats_after_split(dataset):
        train_data, test_data, item2idx, n_items = dataset
        num_train_clicks = sum([len(sample) for sample in train_data])
        num_test_clicks = sum([len(sample) for sample in test_data])
        print(f"num train clicks {num_train_clicks}")
        print(f"num test clicks {num_test_clicks}")
        print(f"num total clicks {num_train_clicks + num_test_clicks}")
        train_data_items = list(chain.from_iterable(train_data))
        test_data_items = list(chain.from_iterable(test_data))
        max_item_index = max(max(train_data_items), max(test_data_items))
        min_item_index = min(min(train_data_items), min(test_data_items))
        print(f"item index range [min_item_index, max_item_index]")
        print(f"{n_items} items")
    
    print("post sequence splitting 1/64 statistics:")
    ds_stats_after_split(dataset_64)

    print("post sequence splitting 1/4 statistics:")
    ds_stats_after_split(dataset_4)

def fix_dataset_samples(dataset):
    """fixes index off by 1 and reverses item2index lookup

    Returns:
        train_data, test_data, idx2item, n_items
    """
    train_data, test_data, item2idx, n_items = dataset

    # removing all unnecessary sample information
    train_data = [sample.items_idxes for sample in train_data.samples]
    test_data = [sample.items_idxes for sample in test_data.samples]

    # subtracting 1 from each index as there's an off by 1
    train_data = [[idx-1 for idx in session] for session in train_data]
    test_data = [[idx-1 for idx in session] for session in test_data]

    # reversing idx -> item store and fixing key index off by 1
    idx2item = {v-1:k for k, v in item2idx.items()}

    return train_data, test_data, idx2item, n_items-1

if __name__ == "__main__":
    assert(input_click_file.exists())
    
    with cwd(STAMP_folder/"datas"/"data"):
        if not Path("yoochoose-clicks.csv").exists():
            print(f"symlinking {input_click_file} to {Path().cwd()/'yoochoose-clicks.csv'}")
            os.symlink(input_click_file, Path().cwd()/"yoochoose-clicks.csv")
        print(f"executing {Path().cwd()/'process_rsc.py'}")
        print("cwd:", Path().cwd())
        from STAMP.datas.data import process_rsc
        balazs_train_file = Path().cwd()/'rsc15_train_full.txt'
        balazs_test_file = Path().cwd()/'rsc15_test.txt'
    
    print(f"generated {balazs_train_file}")
    print(f"generated {balazs_test_file}")

    sys.path.append(str(STAMP_folder.resolve()))
    with cwd(STAMP_folder):
        print(f"executing {Path().cwd()/'data_prepare'/'rsyc15data_read_p.py'} on 1/64")
        from data_prepare.rsyc15data_read_p import load_data_p
        dataset_64 = load_data_p(
            balazs_train_file,
            balazs_test_file,
            pro = 64
        )
    dataset_64 = fix_dataset_samples(dataset_64)

    os.makedirs(save_folder, exist_ok=True)
    save_path = save_folder/"dataset_64.pkl"
    print(f"saving dataset at {save_path}")
    with open(save_path, "wb+") as handle:
        pickle.dump(dataset_64, handle)

    with cwd(STAMP_folder):
        print(f"running {Path().cwd()/'data_prepare'/'rsyc15data_read_p.py'} on 1/4")
        from data_prepare.rsyc15data_read_p import load_data_p
        dataset_4 = load_data_p(
            balazs_train_file,
            balazs_test_file,
            pro = 4
        )
    dataset_4 = fix_dataset_samples(dataset_4)

    save_path = save_folder/"dataset_4.pkl"
    print(f"saving dataset at {save_path}")
    with open(save_path, "wb+") as handle:
        pickle.dump(dataset_4, handle)

    dataset_statistics(process_rsc.train, process_rsc.test, dataset_64, dataset_4)