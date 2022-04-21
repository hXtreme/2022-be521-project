import numpy as np


def fn_line_length(x: np.ndarray):
    return np.sum(np.absolute(np.diff(x, axis=0)), axis=0)


def fn_signed_area(x: np.ndarray):
    return np.sum(x, axis=0)


def fn_area(x: np.ndarray):
    return np.sum(np.absolute(x), axis=0)


def fn_energy(x: np.ndarray):
    return np.linalg.norm(x, axis=0) ** 2


def fn_max_min_diff(x: np.ndarray):
    return np.max(x, axis=0) - np.min(x, axis=0)


def fn_deviation(x: np.ndarray):
    return np.std(x, axis=0)


def fn_avg_peak_to_peak(x: np.ndarray):
    raise NotImplementedError()
    return np.mean(np.max(x, axis=0) - np.min(x, axis=0))


def fn_zero_crossings(x: np.ndarray):
    x_cent = x - np.mean(x, axis=0)
    x_shift = np.hstack([x_cent[:, 1:], x_cent[:, -1].reshape((-1, 1))])
    x_cross = (x_cent * x_shift) < 0
    return np.sum(x_cross, axis=0)
