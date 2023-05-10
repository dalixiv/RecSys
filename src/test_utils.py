from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd


def group_values_by_user(df: pd.DataFrame) -> pd.DataFrame:
    df_ = deepcopy(df)
    df_ = df_.groupby("user")["item"].apply(list).to_frame()
    df_.reset_index(inplace=True)
    return df_


def _topk_recall_(a: List[int], b: List[int]) -> float:
    nom = float(len(set(a).intersection(set(b))))
    denom = float(len(a))
    return nom / denom


def topk_recall(data_pred: pd.DataFrame, data_gt: pd.DataFrame, k: int = 10) -> float:
    data_gt = group_values_by_user(data_pred)

    data_pred.set_index("user", inplace=True)
    data_pred_dict = data_pred.to_dict()

    recall = []
    for raw in data_gt.index:
        user = data_gt["user"][raw]
        items_gt = data_gt["item"][raw]
        items_rec = data_pred_dict["item"][user][:k]

        recall.append(_topk_recall_(items_gt, items_rec))

    recall = np.array(recall)
    return recall.mean(), recall
