from typing import Callable

import numpy as np

from PyANN.functions import relu, relu_d, tanh, tanh_d
from PyANN.utils import add_col, remove_col


class Dense:
    def __init__(self, inputs: int, outputs: int, activation: str = "relu"):
        """
        Inits the "dense" layer type
        """
        self.inputs: int = inputs
        self.outputs: int = outputs

        self.weights: np.ndarray = np.random.normal(
            0,
            (2 * inputs + 2) ** (-0.5),
            (inputs + 1, outputs)
        )

        self.last_step: np.ndarray = None

        if activation.lower() == "relu":
            self.activation: Callable = relu
            self.activation_d: Callable = relu_d
        elif activation.lower() == "tanh":
            self.activation: Callable = tanh
            self.activation_d: Callable = tanh_d

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes a prediction based on an input
        Feeds forward x
        """
        x_with_bias: np.ndarray = add_col(x)
        weighted_sum: np.ndarray = x_with_bias.dot(self.weights)
        prediction: np.ndarray = self.activation(weighted_sum)

        return prediction

    def error(self, error: np.ndarray) -> np.ndarray:
        """
        Backpropagates a network error through this layer
        """
        weight_transpose: np.ndarray = self.weights.T
        error_product: np.ndarray = error.dot(weight_transpose)
        error: np.ndarray = remove_col(error_product)

        return error

    def delta(self, x: np.ndarray, error: np.ndarray) -> np.ndarray:
        """
        Returns the delta that minimises error for the inputs "x"
        """
        x_with_bias: np.ndarray = add_col(x)
        transposed_x: np.ndarray = x_with_bias.T
        layer_output: np.ndarray = self.predict(x)
        gradients: np.ndarray = self.activation_d(layer_output)
        error_gradients: np.ndarray = error * gradients
        delta: np.ndarray = transposed_x.dot(error_gradients)

        return delta

    def apply_delta(self, x: np.ndarray, error: np.ndarray, learning_rate: float, momentum_rate: float = 0):
        """
        Applies the delta that minimises error for the inputs "x"

        Args:
            x: The inputs to the layer to minimise error for
            error: The calculated error to be minimised
            learning_rate: The size of the step to take when adjusting the network weights
            momentum_rate: The amount of momentum to keep from the previous step
        """
        delta: np.ndarray = self.delta(x, error)
        delta_step: np.ndarray = delta * learning_rate

        delta_step_momentum: np.ndarray = delta_step.copy()

        if self.last_step is not None:
            delta_step_momentum = delta_step_momentum + self.last_step * momentum_rate

        self.weights = self.weights + delta_step_momentum

        self.last_step = delta_step
