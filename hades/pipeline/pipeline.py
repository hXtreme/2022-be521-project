from abc import abstractmethod
import numpy as np

import hades


class Pipeline:
    def __init__(self, name, data_dir):
        self.name = name
        self.data_dir = data_dir

    def pre_process_X(self, X: np.ndarray, *args, **kwds):
        return X

    def pre_process_eval_X(self, X: np.ndarray, *args, **kwds):
        return X

    def pre_process_Y(self, Y: np.ndarray, *args, **kwds):
        return Y

    def post_process_Y(self, _X: np.ndarray, _Y: np.ndarray, *args, **kwds):
        return _Y

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray):
        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, subject_id, *args, **kwds) -> np.ndarray:
        train_data, train_label, test_data = hades.utils.load_data(
            self.data_dir, subject_id
        )

        X_train = self.pre_process_X(train_data, *args, **kwds)
        Y_train = self.pre_process_Y(train_label, *args, **kwds)
        self.fit(X_train, Y_train)
        X_test = self.pre_process_eval_X(test_data, *args, **kwds)
        _Y_test = self.predict(X_test)
        Y_test = self.post_process_Y(X_test, _Y_test, *args, **kwds)
        assert len(Y_test) == len(
            test_data
        ), f"Predicted data length ({len(Y_test)}) does not match test data length ({len(test_data)})"
        return Y_test


def run_pipeline(pipeline, save_dir, *args, **kwds):
    result = [pipeline(subject_id, *args, **kwds) for subject_id in range(3)]
    hades.utils.dump_data(f"{save_dir}/leaderboard_{pipeline.name}_preds.mat", *result)
    return result
