import numpy as np
from numpy import vectorize


@vectorize
def relu(x: float) -> float:
    """
    The ReLu activation function

    Args:
        x: The input to the function

    Returns:
        The output of the function
    """
    if x > 0:
        return x
    return 0.1 * x


@vectorize
def relu_d(x: float) -> float:
    """
    The gradient of the ReLu activation function

    Args:
        x: The point at which to take the gradient

    Returns:
        The gradient of the ReLu activation function
    """
    if x > 0:
        return 1
    return 0.1


tanh = np.tanh  # Assigned to this module for code clarity in other files


@vectorize
def tanh_d(x: float) -> float:
    """
    The gradient of the Hyperbolic Tangent activation function

    Args:
        x: The point at which to take the gradient

    Returns:
        The gradient of the tanh activation function
    """
    return 1 - x ** 2
