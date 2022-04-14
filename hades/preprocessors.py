from typing import Tuple, List

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


def filter_data(data, pass_band=(0.1, 100), fs=1000):
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


def exclude_channels(data: np.ndarray, exclude_channels: List) -> np.ndarray:
    """
    Exclude the given channels from the data.
    """
    all_channel_indices = np.arange(data.shape[1])
    keep_channel_indices = np.setdiff1d(all_channel_indices, exclude_channels)
    return data[:, keep_channel_indices]


def create_evolution_matrix(features: np.ndarray, history: int) -> np.ndarray:
    """
    Create the evolution matrix.

    :param features: The features to create the evolution matrix for (windows, channels x features).
    :param history: Number of windows of history to use.
    :return: The evolution matrix (windows, history x channels x features).
    """
    num_windows, num_channels = np.shape(features)
    padded_features = np.vstack((features[: history - 1], features))

    R = np.empty((num_windows, (num_channels * history) + 1))
    R[:, 0] = np.ones((num_windows))
    for i in range(len(padded_features) - 2):
        R[i, 1:] = np.concatenate(
            (
                padded_features[i, :],
                padded_features[i + 1, :],
                padded_features[i + 2, :],
            ),
            axis=None,
        )
    return R


def get_label_clips(labels: np.ndarray) -> List:
    """
    Get the clips for the labels.
    """
    clips = []
    for f_label in labels.T:
        clips.append(f_label[f_label < 0].mean())
    return clips
