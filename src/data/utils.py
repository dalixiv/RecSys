from copy import copy

import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm


def create_user_item_matrix(dataset: pd.DataFrame) -> scipy.sparse.csr_matrix:
    n_users = dataset["user"].max() + 1
    n_items = dataset["item"].max() + 1

    print(f"n_users = {n_users}")
    print(f"n_items = {n_items}")

    m = scipy.sparse.csr_matrix((n_users, n_items), dtype=np.float32).tolil()

    for row in tqdm(dataset.index, desc="filling user-item matrix"):
        index_user = dataset["user"][row]
        index_item = dataset["item"][row]
        rating = dataset["count"][row]
        m[index_user, index_item] = rating

    m = m.tocsr()

    return m


def create_query_user_item_matrix(
    dataset: pd.DataFrame, full_matrix: scipy.sparse.csr_matrix
) -> scipy.sparse.csr_matrix:
    users = dataset[["user"]].groupby("user").sum()
    users.sort_values(by=["user"], inplace=True)
    idxs_users = list(users.index)
    n_users = len(idxs_users)

    print(f"n_users query = {n_users}")
    n_items = full_matrix.shape[1]

    m = scipy.sparse.csr_matrix((n_users, n_items), dtype=np.float32).tolil()

    m = copy(full_matrix[idxs_users])

    return m, users
