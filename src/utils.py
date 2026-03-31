import numpy as np


def round_matrix(arr: np.ndarray) -> np.ndarray:
    return np.where(arr >= 0, np.floor(arr + 0.5), np.ceil(arr - 0.5)).astype(np.int64)


round_matrix(np.array([[-14.75, -1.5, -0.5], [0, 0.5, 1.5]]))
