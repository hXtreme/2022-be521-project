from abc import abstractmethod
from venv import create
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from tqdm import tqdm, trange

from hades import features as available_features

from hades import preprocessors, utils
from hades.pipeline import Pipeline

from hades.pipeline.windowed_feature_pipeline import WindowedFeaturePipeline

class transformerPipeline(WindowedFeaturePipeline):

    NAME = "transformerPipeline"

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

