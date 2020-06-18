import numpy as np


def add_col(matrix: np.array) -> np.array:
    """
    Adds a column of ones to "matrix"
    """
    ones_vector: np.array = np.ones((matrix.shape[0], 1))
    result: np.array = np.append(matrix, ones_vector, axis=1)

    return result


def remove_col(matrix: np.array) -> np.array:
    """
    Removes a column from the end of "matrix"
    """
    result: np.array = matrix[:, :-1]

    return result
