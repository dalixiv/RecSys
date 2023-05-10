import sys

# from surprise import Dataset, Reader
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import scipy

# from surprise import KNNBasic
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from src.data.utils import create_user_item_matrix
from src.recsys import UserBasedCF
from src.similarity import cosine_similarity

# sys.path.append('..')


data_root = Path("./dataset/")


model = UserBasedCF(
    fname_data=data_root / "dataset_indexed_train.csv",
    batch_size=100,
    topk_items=100,
    topk_users=20,
)


result = model.predict(data_root / "dataset_indexed_test.csv")
