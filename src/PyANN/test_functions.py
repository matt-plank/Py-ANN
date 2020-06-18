import unittest

import numpy as np

from PyANN.functions import relu, relu_d


class TestUtils(unittest.TestCase):
    def test_relu(self):
        # ReLu applied to a matrix of positives has no effect at all
        matrix_ones: np.array = np.ones((5, 5))
        matrix_ones_mapped: np.array = relu(matrix_ones)

        self.assertTrue((matrix_ones_mapped == matrix_ones).all())

        # ReLu applied to a matrix of negatives divides everything by 10
        matrix_negatives: np.array = -1 * np.ones((5, 5))
        matrix_negatives_mapped: np.array = relu(matrix_negatives)

        self.assertTrue((matrix_negatives_mapped == matrix_negatives * 0.1).all())

    def test_relu_d(self):
        # ReLu applied to a matrix of positives replaces everything with one
        matrix_twos: np.array = np.ones((5, 5)) * 2
        matrix_twos_mapped: np.array = relu_d(matrix_twos)

        self.assertTrue((matrix_twos_mapped == matrix_twos * 0.5).all())

        # ReLu applied to a matrix of negatives replaces everything with 0.1
        matrix_negatives: np.array = -1 * np.ones((5, 5))
        matrix_negatives_mapped: np.array = relu_d(matrix_negatives)

        self.assertTrue((matrix_negatives_mapped == np.ones_like(matrix_negatives) * 0.1).all())


if __name__ == '__main__':
    unittest.main()
