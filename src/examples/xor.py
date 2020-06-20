import logging

import numpy as np

from PyANN import ANN, Dense

logging.basicConfig(level=logging.INFO)


def main():
    # Initialise the layers for the model
    layer_1: Dense = Dense(2, 4, activation="tanh", momentum_rate=0.5)
    layer_2: Dense = Dense(4, 1, momentum_rate=0.5)

    # Initialise the model we're going to use
    model: ANN = ANN(
        layer_1,
        layer_2
    )

    # Prepare the training data
    xs: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    ys: np.ndarray = np.array([[-1], [1], [1], [-1]])
    print("Target outputs: ")
    print(ys)
    print()

    # Calculate pre-training results
    pre_train_y: np.ndarray = model.predict(xs)
    print("Pre-training Predictions:")
    print(pre_train_y)
    print()

    # Perform training operations
    print("Training..")
    model.train(
        xs,
        ys,
        200,
        0.3
    )
    print()

    # Calculate post-training results
    post_train_y: np.ndarray = model.predict(xs)
    print("Post-training Predictions:")
    print(post_train_y)
    print()


if __name__ == "__main__":
    main()
