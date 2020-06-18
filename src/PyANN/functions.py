from numpy import vectorize


@vectorize
def relu(x: float) -> float:
    """
    The ReLu activation function
    """
    if x > 0:
        return x
    return 0.1 * x


@vectorize
def relu_d(x: float) -> float:
    """
    The gradient of the ReLu activation function
    """
    if x > 0:
        return 1
    return 0.1
