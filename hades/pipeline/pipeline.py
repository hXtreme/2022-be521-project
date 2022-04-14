from abc import abstractmethod
import numpy as np

from tqdm import tqdm, trange

import hades
from hades import preprocessors, postprocessors


class Pipeline:
    def __init__(self, name):
        self.name = name

    def pre_process_X(self, X: np.ndarray, *args, **kwds):
        return X

    def pre_process_eval_X(self, X: np.ndarray, *args, **kwds):
        return X

    def pre_process_Y(self, Y: np.ndarray, *args, **kwds):
        return Y

    def post_process_Y(self, _X: np.ndarray, _Y: np.ndarray, *args, **kwds):
        return _Y

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X = self.pre_process_X(X)
        Y = self.pre_process_Y(Y)
        return self._fit(X, Y)

    @abstractmethod
    def _fit(self, X: np.ndarray, Y: np.ndarray):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        _X = self.pre_process_eval_X(X)
        _Y = self._predict(_X)
        Y = self.post_process_Y(_X, _Y)
        assert len(Y) == len(
            X
        ), f"Predicted data length ({len(Y)}) does not match test data length ({len(X)})"
        return Y

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, train_data, train_label, test_data, *args, **kwds) -> np.ndarray:
        self.fit(train_data, train_label)
        Y_test = self.predict(test_data)
        return Y_test


def run_pipeline(
    pipeline,
    data_dir,
    save_dir,
    dev_split=None,
    filter_data=None,
    pred_clipping=False,
    *args,
    **kwds,
):
    result = []
    for subject_id in trange(3, desc=" Subjects ", position=0):
        train_data_full, train_label_full, test_data = hades.utils.load_data(
            data_dir, subject_id
        )
        if filter_data is not None:
            train_data_full = preprocessors.filter_data(
                train_data_full, pass_band=filter_data, fs=1000
            )
            test_data = preprocessors.filter_data(
                test_data, pass_band=filter_data, fs=1000
            )
        train_data, train_label, dev_data, dev_label = hades.utils.split_data(
            train_data_full, train_label_full, dev_split
        )
        preds = pipeline(train_data, train_label, test_data, *args, **kwds)
        if pred_clipping:
            pred_clips = preprocessors.get_label_clips(train_label)
            preds = postprocessors.clip_predictions(preds, pred_clips)

        result.append(preds)

    hades.utils.dump_data(
        f"{save_dir}/leaderboard_{pipeline.name}{'' if not filter_data else '_filt' + str(filter_data)}{'' if not pred_clipping else '_clip'}_preds.mat",
        *result,
    )
    return result
