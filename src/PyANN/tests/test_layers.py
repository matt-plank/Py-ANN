import unittest

import numpy as np

from PyANN.layers import Dense


class TestLayers(unittest.TestCase):
    def test_Dense_predict(self):
        layer: Dense = Dense(3, 22)
        xs: np.ndarray = np.random.random((20, 3))
        ys: np.ndarray = layer.predict(xs)

        self.assertTrue(ys.shape == (xs.shape[0], 22))  # Number of rows is preserved (each row is a datapoint)
        self.assertTrue(len(ys.shape) == len(xs.shape))  # Output has the same number of demensions as

    def test_Dense_predict_momentum(self):
        layer: Dense = Dense(3, 22, momentum_rate=0.5)
        xs: np.ndarray = np.random.random((20, 3))
        ys: np.ndarray = layer.predict(xs)

        self.assertTrue(ys.shape == (xs.shape[0], 22))  # Number of rows is preserved (each row is a datapoint)
        self.assertTrue(len(ys.shape) == len(xs.shape))  # Output has the same number of demensions as input

    def test_Dense_error(self):
        layer: Dense = Dense(2, 1)
        xs: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        ys: np.ndarray = layer.predict(xs)
        desired_ys: np.ndarray = np.array([[0], [1], [1], [0]])
        network_error: np.ndarray = desired_ys - ys
        error: np.ndarray = layer.error(network_error)

        self.assertTrue(len(error.shape) == len(network_error.shape))  # The error matrix doesn't change dimensions
        self.assertTrue(error.shape == xs.shape)  # There is an error value for every input to the layer

    def test_Dense_delta(self):
        layer: Dense = Dense(2, 1)
        xs: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        ys: np.ndarray = layer.predict(xs)
        desired_ys: np.ndarray = np.array([[0], [1], [1], [0]])
        network_error: np.ndarray = desired_ys - ys
        delta: np.ndarray = layer.delta(xs, network_error)

        self.assertTrue(
            delta.shape == layer.weights.shape)  # The layer weight delta must be the same shape as the layer weights

    def test_Dense_apply_delta(self):
        np.random.seed(100)

        layer: Dense = Dense(2, 1)
        xs: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        ys: np.ndarray = layer.predict(xs)
        desired_ys: np.ndarray = np.array([[0], [1], [1], [1]])
        network_error: np.ndarray = desired_ys - ys

        layer.apply_delta(xs, network_error, 0.001)

        ys_recaltulated: np.ndarray = layer.predict(xs)
        error_recalcualted: np.ndarray = desired_ys - ys_recaltulated

        # Applying a delta should reduce error
        # This requires a seed because sometimes the step is too large
        self.assertTrue(error_recalcualted.sum() <= network_error.sum())  # Applying a delta should always reduce error

    def test_Dense_apply_delta_momentum(self):
        np.random.seed(100)

        layer: Dense = Dense(2, 1, momentum_rate=0.5)
        xs: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        ys: np.ndarray = layer.predict(xs)
        desired_ys: np.ndarray = np.array([[0], [1], [1], [1]])
        network_error: np.ndarray = desired_ys - ys

        layer.apply_delta(xs, network_error, 0.001)

        ys_recaltulated: np.ndarray = layer.predict(xs)
        error_recalcualted: np.ndarray = desired_ys - ys_recaltulated

        # Applying a delta should reduce error
        # This requires a seed because sometimes the step is too large
        self.assertTrue(error_recalcualted.sum() <= network_error.sum())  # Applying a delta should always reduce error


if __name__ == '__main__':
    unittest.main()
