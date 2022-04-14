from typing import Tuple
import numpy as np
from scipy.io import loadmat, savemat

from scipy import interpolate


def load_data(data_dir, subject_id):
    """
    Loads the training and testing data from the given dir.

    :param data_dir: The directory containing the data.
    :param subject_id: The subject id of the data to load (0, 1, 2).
    :return: A tuple of (training_data, training_labels, testing_data).
    """
    train_path = data_dir + "/raw_training_data.mat"
    test_path = data_dir + "/leaderboard_data.mat"
    train_data = loadmat(train_path)
    test_data = loadmat(test_path)
    return (
        train_data["train_ecog"][subject_id, 0],
        train_data["train_dg"][subject_id, 0],
        test_data["leaderboard_ecog"][subject_id, 0],
    )


def dump_data(path, *dg):
    """
    Dumps the leaderboard data to the given path.

    :param path: The path to dump the data to.
    :param dg: The data to dump.
    """
    temp = np.array([None] * len(dg), dtype=object)
    temp.shape = (len(dg), 1)
    for i, d in enumerate(dg):
        temp[i, 0] = d
    leaderboard_data = {
        "predicted_dg": temp,
    }
    savemat(path, leaderboard_data)


def resample_data(
    data: np.ndarray, fs_old: float, fs_new: float, adjust=False
) -> np.ndarray:
    """
    Resamples the data to the new sampling rate.

    :param data: The data to resample.
    :param fs_old: The old sampling rate.
    :param fs_new: The new sampling rate.
    """
    if fs_old == fs_new:
        return data
    samples = data.shape[-1]
    duration = samples / fs_old
    time_old = np.linspace(0, duration, samples)

    interp_fn = interpolate.interp1d(time_old, data, kind="cubic")
    if fs_new < fs_old:
        samples_new = int((samples * fs_new) / fs_old)
        time_new = np.linspace(0, duration, samples_new)
        resampled_data = interp_fn(time_new[:-1])
    else:
        samples_new = int((samples * fs_new) / fs_old)
        time_new = np.linspace(0, duration, samples_new)
        resampled_data = interp_fn(time_new)
    return resampled_data


def get_time_slice(start, duartion, fs):
    time_slice = np.arange(0, int(duartion * fs)) + int(start * fs)
    return time_slice


def split_data(
    data: np.ndarray, label: np.ndarray, split: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split is None:
        return data, label, data, label
    train_data = data[:split]
    train_label = label[:split]
    dev_data = data[split:]
    dev_label = label[split:]
    return train_data, train_label, dev_data, dev_label


def window_borders(x: np.ndarray, fs: float, win_len: float, win_disp: float):
    assert fs * win_len == int(fs * win_len)
    assert fs * win_disp == int(fs * win_disp)
    win_samples_len = int(fs * win_len)
    win_disp_samples_len = int(fs * win_disp)
    ends = np.flip(np.arange(len(x), win_samples_len - 1, -win_disp_samples_len))
    starts = ends - win_samples_len
    return starts, ends


def windowed_fn(x: np.ndarray, fs: float, win_len: float, win_disp: float, fn):
    starts, ends = window_borders(x, fs, win_len, win_disp)
    for start, end in zip(starts, ends):
        yield fn(x[start:end])
    return None
