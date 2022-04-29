from abc import abstractmethod

import pickle
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from tqdm import tqdm, trange

from hades import features as available_features

from hades import preprocessors, utils
from hades.pipeline import Pipeline


class WindowedFeaturePipeline(Pipeline):
    def __init__(self, name, fs, window_length, window_displacement, history):
        super().__init__(name)
        self.fs = fs
        self.window_length = window_length
        self.window_displacement = window_displacement
        self.history = history
        self.feature_fs = 1 / self.window_displacement
        self.magic = int((window_length / window_displacement) - 1)
        self.sid = None
        assert self.feature_fs == int(
            self.feature_fs
        ), "Feature frequency must be an integer"

    def get_windowed_feats(self, X: np.ndarray, fs: int, win_len: int, win_disp: int):
        return np.array(
            [
                feat
                for feat in utils.windowed_fn(
                    x=X, fs=fs, win_len=win_len, win_disp=win_disp, fn=self.get_features
                )
            ]
        )

    def get_features(self, window: np.ndarray, *args, **kwds):
        """
        Get features from a window of data

        :param window: window of data (window_samples, channels)
        :return: features (channels x num_features,)
        """
        retval = np.array([fn(window) for fn in self.features]).T
        samples, channels = window.shape
        channels1, num_features = retval.shape
        assert channels == channels1 and num_features == len(self.features)
        return retval.reshape((-1,))

    def pre_process_X(self, X: np.ndarray, *args, **kwds):
        X, self.mean, self.std = preprocessors.normalize(X)
        features = self.get_windowed_feats(
            X, fs=self.fs, win_len=self.window_length, win_disp=self.window_displacement
        )
        features = self.features_hook(features)
        features_evolution_matrix = preprocessors.create_evolution_matrix(
            features, history=self.history
        )
        return features_evolution_matrix

    def pre_process_eval_X(self, X: np.ndarray, *args, **kwds):
        X, _, _ = preprocessors.normalize(X, mean=self.mean, std=self.std)
        features = self.get_windowed_feats(
            X, fs=self.fs, win_len=self.window_length, win_disp=self.window_displacement
        )
        features = self.features_hook_eval(features)
        features_evolution_matrix = preprocessors.create_evolution_matrix(
            features, history=self.history
        )
        return features_evolution_matrix

    def pre_process_Y(self, Y: np.ndarray, *args, **kwds):
        return utils.resample_data(
            Y.T, fs_old=self.fs, fs_new=self.feature_fs, magic=self.magic
        ).T

    def post_process_Y(self, X: np.ndarray, Y: np.ndarray, *args, **kwds):
        last_Y = Y[-self.magic :]
        Y = np.vstack((Y, last_Y))
        return utils.resample_data(Y.T, fs_old=self.feature_fs, fs_new=self.fs).T

    def features_hook(self, features):
        """
        Hook to modify features before they are used in the pipeline

        :param features: features (windows, channels x features)
        """
        return features

    def features_hook_eval(self, features):
        """
        Hook to modify features before they are used in the pipeline

        :param features: features (windows, channels x features)
        """
        return features


class Part1(WindowedFeaturePipeline):
    NAME = "part1_pipeline"

    def __init__(self, fs, window_length=100e-3, window_displacement=50e-3, history=3):
        super().__init__(
            f"{self.NAME}_wl{window_length}_wd{window_displacement}_h{history}",
            fs,
            window_length,
            window_displacement,
            history,
        )

    @property
    def features(self):
        return [
            available_features.fn_line_length,
            available_features.fn_area,
            available_features.fn_energy,
            available_features.fn_signed_area,
        ]

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        self.model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, Y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class WindowedKNN(WindowedFeaturePipeline):
    NAME = "windowed_knn_pipeline"

    def __init__(
        self, fs, window_length=100e-3, window_displacement=50e-3, history=3, k=3
    ):
        name = f"{self.NAME}_{k}-NN"
        super().__init__(
            f"{name}_wl{window_length}_wd{window_displacement}_h{history}",
            fs,
            window_length,
            window_displacement,
            history,
        )
        self.k = k

    @property
    def features(self):
        return [
            available_features.fn_line_length,
            available_features.fn_area,
            available_features.fn_energy,
            available_features.fn_signed_area,
        ]

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        self.model = make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=self.k)
        ).fit(X, Y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class Part1MLP(WindowedFeaturePipeline):
    NAME = "part1_mlp_pipeline"

    def __init__(
        self,
        fs,
        window_length=100e-3,
        window_displacement=50e-3,
        history=3,
        layers=(100,),
    ):
        super().__init__(
            f"{self.NAME}_wl{window_length}_wd{window_displacement}_h{history}_l{layers}",
            fs,
            window_length,
            window_displacement,
            history,
        )
        self.layers = layers

    @property
    def features(self):
        return [
            available_features.fn_line_length,
            available_features.fn_area,
            available_features.fn_energy,
            available_features.fn_signed_area,
        ]

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        self.model = make_pipeline(
            StandardScaler(), MLPRegressor(hidden_layer_sizes=self.layers)
        ).fit(X, Y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class MLP2(WindowedFeaturePipeline):
    NAME = "mlp2_pipeline"

    def __init__(
        self,
        fs,
        window_length=100e-3,
        window_displacement=50e-3,
        history=3,
        layers=(100,),
    ):
        super().__init__(
            f"{self.NAME}_wl{window_length}_wd{window_displacement}_h{history}_l{layers}",
            fs,
            window_length,
            window_displacement,
            history,
        )
        self.name += f"_fts{len(self.features)}"
        self.layers = layers

    @property
    def features(self):
        return [
            available_features.fn_line_length,
            available_features.fn_area,
            available_features.fn_energy,
            available_features.fn_signed_area,
            available_features.fn_deviation,
            # available_features.fn_max_min_diff,
        ]

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        self.model = make_pipeline(
            StandardScaler(), MLPRegressor(hidden_layer_sizes=self.layers)
        ).fit(X, Y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class MLP2_Fingers(WindowedFeaturePipeline):
    NAME = "mlp2_fingers_pipeline"

    def __init__(
        self,
        fs,
        window_length=100e-3,
        window_displacement=50e-3,
        history=3,
        layers=(100,),
    ):
        super().__init__(
            f"{self.NAME}_wl{window_length}_wd{window_displacement}_h{history}_l{layers}",
            fs,
            window_length,
            window_displacement,
            history,
        )
        self.name += f"_fts{len(self.features)}"
        self.layers = layers

    @property
    def features(self):
        return [
            available_features.fn_line_length,
            available_features.fn_area,
            available_features.fn_energy,
            available_features.fn_signed_area,
            available_features.fn_deviation,
            # available_features.fn_max_min_diff,
        ]

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        self.models = []

        for _Y in Y.T:
            model = make_pipeline(
                StandardScaler(), MLPRegressor(hidden_layer_sizes=self.layers)
            ).fit(X, _Y)
            self.models.append(model)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        Y = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models):
            Y[:, i] = model.predict(X)
        return Y
