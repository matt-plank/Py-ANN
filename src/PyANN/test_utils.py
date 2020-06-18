import unittest

import numpy as np

from PyANN.utils import add_col, remove_col


class TestUtils(unittest.TestCase):
    def test_add_col(self):
        matrix: np.array = np.ones((5, 5))

        matrix_with_col: np.array = add_col(matrix)

        self.assertTrue(matrix.shape[0] == matrix_with_col.shape[0])
        self.assertTrue(matrix.shape[1] == matrix_with_col.shape[1] - 1)

    def test_remove_col(self):
        matrix: np.array = np.ones((5, 5))

        matrix_without_col: np.array = remove_col(matrix)

        self.assertTrue(matrix.shape[0] == matrix_without_col.shape[0])
        self.assertTrue(matrix.shape[1] - 1 == matrix_without_col.shape[1])


if __name__ == '__main__':
    unittest.main()
