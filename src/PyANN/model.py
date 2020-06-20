from typing import List

from PyANN.layers import Dense

import numpy as np


class ANN:
    def __init__(self, *layers):
        """
        Initialises the ANN class with a set of layers

        Args:
            *layers: The layers to load the model with
        """
        self.layers: List[Dense] = layers
        self.node_layers: List[np.ndarray] = []

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Feeds-forwards some inputs through the network to give a prediction

        Args:
            x: The inputs to feed-forward

        Returns:
            The model's prediction from the inputs
        """
        self.node_layers = [x]
        for layer in self.layers:
            next_layer = layer.predict(self.node_layers[-1])
            self.node_layers.append(next_layer)

        return self.node_layers[-1]

    def train(self, xs: np.ndarray, desired_ys: np.ndarray, epochs: int, learning_rate: float):
        """
        Trains the neural network to match inputs "xs" to inputs "ys"

        Args:
            xs: The input set
            desired_ys: The output set to match the input set to
            epochs: The number of iterations to train through
            learning_rate: The size of the steps to take at each learning stage
        """
        for _ in range(epochs):
            ys: np.ndarray = self.predict(xs)

            errors: List[np.ndarray] = [desired_ys - ys]

            for layer in self.layers[::-1][:-1]:
                layer_error: np.ndarray = layer.error(errors[-1])
                errors.insert(0, layer_error)

            for i, layer in enumerate(self.layers):
                self.layers[i].apply_delta(self.node_layers[i], errors[i], learning_rate)
