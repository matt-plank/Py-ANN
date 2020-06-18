from typing import Callable

import numpy as np

from PyANN.functions import relu, relu_d
from PyANN.utils import add_col, remove_col


class Dense:
    def __init__(self, inputs: int, outputs: int, activation: str = "relu"):
        """
        Inits the "dense" layer type
        """
        self.inputs: int = inputs
        self.outputs: int = outputs

        self.weights: np.array = np.random.normal(
            0,
            (2 * inputs + 2) ** (-0.5),
            (inputs + 1, outputs)
        )

        if activation.lower() == "relu":
            self.activation: Callable = relu
            self.activation_d: Callable = relu_d

    def predict(self, x: np.array) -> np.array:
        """
        Makes a prediction based on an input
        Feeds forward x
        """
        x_with_bias: np.array = add_col(x)
        weighted_sum: np.array = x_with_bias.dot(self.weights)
        prediction: np.array = self.activation(weighted_sum)

        return prediction

    def error(self, error: float) -> np.array:
        """
        Backpropagates a network error through this layer
        """
        return remove_col(error.dot(self.weights.T))

    def delta(self, x: np.array, error: np.array) -> np.array:
        """
        Returns the delta that minimises error for the inputs "x"
        """
        x_with_bias:     np.array = add_col(x)
        transposed_x:    np.array = x_with_bias.T
        layer_output:    np.array = self.predict(x)
        gradients:       np.array = self.activation_d(layer_output)
        error_gradients: np.array = error * gradients
        delta:           np.array = transposed_x.dot(error_gradients)

        return delta

    def apply_delta(self, x: np.array, error: np.array, learning_rate: float):
        """
        Applies the delta that minimises error for the inputs "x"
        """
        delta: np.array = self.delta(x, error)
        delta_step: np.array = delta * learning_rate

        self.weights = self.weights + delta_step
