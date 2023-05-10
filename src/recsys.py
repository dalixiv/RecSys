from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

from .data.utils import create_query_user_item_matrix, create_user_item_matrix
from .similarity import cosine_similarity


class UserBasedCF(object):
    def __init__(
        self,
        fname_data: Union[Path, str],
        topk_items: int = 20,
        topk_users: int = 20,
        batch_size=1000,
    ) -> None:

        fname_data = Path(fname_data)
        self.dataset_df = pd.read_csv(fname_data)
        self.user_item_matrix = create_user_item_matrix(self.dataset_df)
        self.batch_size = batch_size
        self.topk_u = topk_users
        self.topk_i = topk_items

    def predict(
        self,
        fname: Union[str, Path],
        topk_u: Optional[int] = None,
        topk_i: Optional[int] = None,
        weighted: bool = False,
    ) -> None:

        fname = Path(fname)

        topk_u = topk_u if topk_u is not None else self.topk_u
        topk_i = topk_i if topk_i is not None else self.topk_i

        dataset_query_df = pd.read_csv(fname)
        # user_names_df = dataset_query_df[['user']].groupby('user').sum()
        # user_names_df.reset_index(inplace=True)

        user_item_query_matrix, user_names_df = create_query_user_item_matrix(
            dataset_query_df, self.user_item_matrix
        )

        cs = cosine_similarity(
            self.user_item_matrix, user_item_query_matrix, batch_size=self.batch_size
        )

        print("computing similar users ...")
        kth = cs.shape[0] - topk_u
        indexes = np.argpartition(cs, kth, axis=0)
        similar_users_scores = np.hstack(
            [cs[indexes[:, l]][:, l : l + 1] for l in range(indexes.shape[-1])]
        )[-topk_u:]

        indexes_similar_users = indexes[-topk_u:]  # (topk, n_test_users)

        result = {"user": [], "item": [], "score": []}

        for i in tqdm(
            range(user_item_query_matrix.shape[0]),
            desc="creation recomendations for each in query",
        ):
            result["user"].append(user_names_df.index[i])
            indexes_zero = user_item_query_matrix[i].toarray()
            indexes_zero = np.where(indexes_zero == 0)[0]
            idxs, score = self.predict_for_one_user(
                indexes_similar_users[:, i], topk_i, indexes_zero
            )
            idxs_score = [
                [idx, s]
                for idx, s in sorted(zip(idxs, score), key=lambda x: x[1], reverse=True)
            ]
            result["item"].append([e[0] for e in idxs_score])
            result["score"].append([e[1] for e in idxs_score])

        result_df = pd.DataFrame(result)
        res_fname = Path(fname.parent) / ("rec_" + fname.name)

        result_df.to_csv(res_fname, index=True)
        return result_df

    def predict_for_one_user(
        self,
        similar_users_idxs: np.ndarray,
        topk_i: Optional[int] = None,
        indexes_zero: np.ndarray = np.array([]),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        make prediction for one user by its similar users -> indexes of non-zero elements and its scores
        """
        topk_i = topk_i if topk_i is not None else self.topk_i

        similar_users_items_matrix = self.user_item_matrix[similar_users_idxs].toarray()
        scores = similar_users_items_matrix.mean(axis=0)[indexes_zero]

        indexes = np.argpartition(scores, scores.shape[0] - topk_i)[-topk_i:]
        return (indexes, scores[indexes])
