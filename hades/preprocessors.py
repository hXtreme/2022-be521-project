from typing import Tuple

import numpy as np

from scipy import signal as sig


def normalize(
    data: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std


def filter_data(data, pass_band=(0.1, 200), fs=1000):
    """
    Write a filter function to clean underlying data.
    Filter type and parameters are up to you. Points will be awarded for reasonable filter type, parameters and application.
    Please note there are many acceptable answers, but make sure you aren't throwing out crucial data or adversly
    distorting the underlying data!

    Input:
      data (samples x channels): the noisy signal
      fs: the sampling rate (1000 for this dataset)
    Output:
      clean_data (samples x channels): the filtered signal
    """
    sos = sig.butter(10, Wn=pass_band, btype="bandpass", fs=fs, output="sos")
    if len(data.shape) > 1:
        clean_data = np.array([sig.sosfiltfilt(sos, eeg) for eeg in data.T]).T
    else:
        clean_data = sig.sosfiltfilt(sos, data)

    assert clean_data.shape == data.shape
    return clean_data
