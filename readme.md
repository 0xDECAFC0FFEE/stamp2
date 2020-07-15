# readme

1. download dataset from https://www.kaggle.com/chadgostopp/recsys-challenge-2015
2. unzip yoochoose-clicks.dat into data/raw/yoochoose
3. run `python3 src/features/build_features.py`
    - folder containing unmodified STAMP code in src/features/STAMP
    - folder containing unmodified process_rsc.py from "Session-based Recommendations with Recurrent Neural Networks" in src/features/STAMP/datas/data
4. docker build -t stamp2_image .
5. nvidia-docker run -it --shm-size=1g --ulimit memlock=-1 -e "TERM=xterm-256color" -v $(pwd):"/workspace/STAMP2" -p 8889:8888 -p 16006:6006 stamp2_image

- models are in src/models
- main notebook is notebooks/stamp2.ipynb