from typing import Tuple
import numpy as np
from scipy.io import loadmat, savemat


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
