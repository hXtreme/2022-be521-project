from typing import overload
import numpy as np

import hades
from hades.pipeline import Pipeline


class SillyPipeline(Pipeline):
    NAME = "silly_pipeline"

    def __init__(self, data_dir, fs):
        super().__init__(self.NAME, data_dir)
        self.fs = fs

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.Y = Y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        time_slice = hades.utils.get_time_slice(1, len(X) / self.fs, fs=self.fs)
        pred = self.Y[time_slice]
        return pred
