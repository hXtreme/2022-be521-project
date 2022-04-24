from abc import abstractmethod
from pathlib import Path
import pickle
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
    preds_jagged=False,
    translate_labels=0,
    dump_model=True,
    *args,
    **kwds,
):
    dump_name = (
        f"leaderboard_{pipeline.name}"
        + (f"_filt{str(filter_data)}" if filter_data else "")
        + (f"_jagg" if preds_jagged else "")
        + (f"_clip" if pred_clipping else "")
        + (f"_tr{translate_labels}" if translate_labels > 0 else "")
    )
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

        if dump_model or True:
            model_dir = Path(f"./models/{dump_name}")
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(f"{model_dir}/model-sub{subject_id}.pkl", "wb") as f:
                pickle.dump(pipeline, f)
        if pred_clipping:
            pred_clips = preprocessors.get_label_clips(train_label)
            preds = postprocessors.clip_predictions(preds, pred_clips)
        if translate_labels > 0:
            preds = postprocessors.translate(preds, translate_labels)
        if preds_jagged:
            preds = postprocessors.jagged(preds, r=20)

        result.append(preds)

    hades.utils.dump_data(f"{save_dir}/{dump_name}_preds.mat", *result)
    return result


def load_pipeline(pipeline_file: str) -> Pipeline:
    with open(pipeline_file, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline
