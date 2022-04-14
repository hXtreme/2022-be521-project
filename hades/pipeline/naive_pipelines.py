import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from tqdm import tqdm, trange

from hades import preprocessors, utils
from hades.pipeline import Pipeline


class Linear(Pipeline):
    NAME = "naive_linear_pipeline"

    def __init__(self, fs):
        super().__init__(self.NAME)
        self.fs = fs

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        self.model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, Y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class SVM(Pipeline):
    NAME = "naive_svm_pipeline"

    def __init__(self, fs):
        super().__init__(self.NAME)
        self.fs = fs

    def pre_process_Y(self, Y: np.ndarray, *args, **kwds):
        return Y.T

    def post_process_Y(self, _X: np.ndarray, _Y: np.ndarray, *args, **kwds):
        return _Y.T

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        self.models = [make_pipeline(StandardScaler(), SVR()).fit(X, y) for y in Y]
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([model.predict(X) for model in self.models])
        return preds


class Lookup(Pipeline):
    NAME = "naive_lookup_pipeline"

    def __init__(self, fs, distance_fn, downsample_factor=20):
        super().__init__(self.NAME)
        self.fs = fs
        self.distance_fn = distance_fn
        self.dsf = downsample_factor

    def pre_process_X(self, X: np.ndarray, *args, **kwds):
        _X, self.X_mean, self.X_std = preprocessors.normalize(X)
        _X = _X.reshape((-1, self.dsf, X.shape[1]))
        return _X

    def pre_process_eval_X(self, X: np.ndarray, *args, **kwds):
        _X, _, _ = preprocessors.normalize(X, mean=self.X_mean, std=self.X_std)
        _X = _X.reshape((-1, self.dsf, X.shape[1]))
        return _X

    def pre_process_Y(self, Y: np.ndarray, *args, **kwds):
        return Y[:: self.dsf, :]

    def post_process_Y(self, _X: np.ndarray, _Y: np.ndarray, *args, **kwds):
        Y = np.tile(_Y, (1, self.dsf)).reshape((self.dsf * _Y.shape[0], _Y.shape[1]))
        return Y

    def __find_closest(self, x):
        closest_distance = np.inf
        closest_idx = 0
        for idx in range(len(self.x)):
            distance = self.distance_fn(x, self.x[idx])
            if distance < closest_distance:
                closest_distance = distance
                closest_idx = idx
        return closest_idx

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        self.x = X
        self.lookup_table_y = dict()
        for idx, y in enumerate(Y):
            self.lookup_table_y[idx] = y
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        ret_val = []
        for idx in trange(len(X), desc=" Predicting ", position=1):
            ret_val.append(self.lookup_table_y[self.__find_closest(X[idx])])
        return np.array(ret_val)


class DownsampledLinear(Pipeline):
    NAME = "downsampled_linear_pipeline"

    def __init__(self, fs, downsample_factor: int = 40):
        super().__init__(self.NAME)
        self.fs = fs
        self.dsf = downsample_factor

    def pre_process_X(self, X: np.ndarray, *args, **kwds):
        _X, self.X_mean, self.X_std = preprocessors.normalize(X)
        _X = _X.reshape((X.shape[0] // self.dsf, -1))
        return _X

    def pre_process_eval_X(self, X: np.ndarray, *args, **kwds):
        _X, _, _ = preprocessors.normalize(X, mean=self.X_mean, std=self.X_std)
        _X = _X.reshape((X.shape[0] // self.dsf, -1))
        return _X

    def pre_process_Y(self, Y: np.ndarray, *args, **kwds):
        return Y[:: self.dsf, :]

    def post_process_Y(self, _X: np.ndarray, _Y: np.ndarray, *args, **kwds):
        Y = np.tile(_Y, (1, self.dsf)).reshape((self.dsf * _Y.shape[0], _Y.shape[1]))
        return Y

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        self.model = LinearRegression().fit(X, Y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
