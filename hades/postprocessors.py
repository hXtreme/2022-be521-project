from typing import List

import numpy as np


def clip_predictions(preds: np.ndarray, pred_clips: List) -> np.ndarray:
    clipped_preds = []
    for fid, f_pred in enumerate(preds.T):
        clipped = np.copy(f_pred)
        clipped[f_pred < 0] = pred_clips[fid]
        clipped_preds.append(clipped)
    clipped_preds = np.array(clipped_preds).T
    assert clipped_preds.shape == preds.shape
    return clipped_preds


def jagged(y, r):
    y1 = y[::r, :]
    res = np.tile(y1, (1, r)).reshape(y.shape)
    return res
