import numpy as np


def round_matrix(arr: np.ndarray) -> np.ndarray:
    return np.where(arr >= 0, np.floor(arr + 0.5), np.ceil(arr - 0.5)).astype(np.int64)
