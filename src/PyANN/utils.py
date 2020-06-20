import numpy as np


def add_col(matrix: np.ndarray) -> np.ndarray:
    """
    Adds a column of ones to a matrix

    Args:
        matrix: The matrix to add the ones to

    Returns:
        "matrix" but with an additional column full of ones
    """
    ones_vector : np.ndarray = np.ones((matrix.shape[0], 1))
    result : np.ndarray = np.append(matrix, ones_vector, axis=1)

    return result


def remove_col(matrix: np.ndarray) -> np.ndarray:
    """
    Removes a column from the end of "matrix"

    Args:
        matrix: The matrix used

    Returns:
        "matrix" without it's last column
    """
    result : np.ndarray = matrix[:, :-1]

    return result
