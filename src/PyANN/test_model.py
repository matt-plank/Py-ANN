import unittest
from typing import List, Tuple

import numpy as np

from PyANN.layers import Dense
from PyANN.model import ANN


class TestModel(unittest.TestCase):
    def test_model_predict(self):
        model: ANN = ANN(
            Dense(2, 4),
            Dense(4, 1)
        )

        xs: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        ys: np.ndarray = model.predict(xs)

        self.assertTrue(ys.shape == (xs.shape[0], 1))

    def test_model_train(self):
        model: ANN = ANN(
            Dense(2, 8, activation="relu"),
            Dense(8, 1, activation="tanh")
        )

        xs: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        desired_ys: np.ndarray = np.array([[-1], [1], [1], [-1]])

        pre_train_y: np.ndarray = model.predict(xs)
        pre_train_shapes: List[Tuple[int, int]] = [layer.weights.shape for layer in model.layers]

        model.train(
            xs,
            desired_ys,
            1000,
            0.1
        )

        post_train_shapes: List[Tuple[int, int]] = [layer.weights.shape for layer in model.layers]
        post_train_y: np.ndarray = model.predict(xs)

        pre_train_error: np.ndarray = desired_ys - pre_train_y
        post_train_error: np.ndarray = desired_ys - post_train_y

        print(pre_train_y)
        print(post_train_y)
        self.assertTrue(abs(post_train_error).mean() < abs(pre_train_error).mean())  # The error must decrease after training
        self.assertListEqual(pre_train_shapes, post_train_shapes)  # The layer weights must stay the same shape through training


if __name__ == '__main__':
    unittest.main()
