import numpy as np
import scipy
from scipy.spatial.distance import cdist
from tqdm import tqdm


def cosine_similarity(
    a: scipy.sparse.csr_matrix,
    b: scipy.sparse.csr_matrix,
    batch_size=1,
) -> scipy.sparse.csr_matrix:

    na, _ = a.shape
    nb, _ = b.shape

    res = np.zeros((na, nb))

    for i in tqdm(range(0, na, batch_size)):
        for j in range(0, nb, batch_size):
            a_arr = a[i : i + batch_size].toarray()
            b_arr = b[j : j + batch_size].toarray()
            dist = -cdist(a_arr, b_arr, "cosine") + 1
            # print(f'{dist.shape}, {res[i, i+batch_size].shape}')
            res[i : i + batch_size, j : j + batch_size] = dist

    return res
