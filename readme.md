# readme

1. add conda packages from requirements.txt
2. download dataset from https://www.kaggle.com/chadgostopp/recsys-challenge-2015
3. unzip yoochoose-clicks.dat into data/raw/yoochoose
4. run `python3 src/features/build_features.py`
    - folder containing unmodified STAMP code in src/features/STAMP
    - folder containing unmodified process_rsc.py from "Session-based Recommendations with Recurrent Neural Networks" in src/features/STAMP/datas/data

- models are in src/models
- main notebook is notebooks/stamp2.ipynb