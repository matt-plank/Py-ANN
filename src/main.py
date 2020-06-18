import numpy as np

from PyANN import Dense

if __name__ == "__main__":
    layer: Dense = Dense(2, 1)
    xs: np.array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    y: np.array = layer.predict(xs)

    print(y)
