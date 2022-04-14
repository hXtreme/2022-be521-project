from typing import overload
import numpy as np

import hades
from hades.pipeline import Pipeline


class SillyPipeline(Pipeline):
    NAME = "silly_pipeline"

    def __init__(self, fs):
        super().__init__(self.NAME)
        self.fs = fs

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        self.Y = Y
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        time_slice = hades.utils.get_time_slice(1, len(X) / self.fs, fs=self.fs)
        pred = self.Y[time_slice]
        return pred
