import numpy as np

from PyANN import Dense, ANN

if __name__ == "__main__":
    # Initialise the layers for the model
    layer_1: Dense = Dense(2, 4, activation="tanh")
    layer_2: Dense = Dense(4, 1)

    # Initialise the model we're going to use
    model: ANN = ANN(
        layer_1,
        layer_2
    )

    # Prepare the training data
    xs: np.ndarray = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    ys: np.ndarray = np.array([[-1], [1], [1], [-1]])

    # Calculate pre-training results
    pre_train_y: np.ndarray = model.predict(xs)

    # Perform training operations
    model.train(
        xs,
        ys,
        300,
        0.2,
        momentum_rate = 0.1
    )

    # Calculate post-training results
    post_train_y: np.ndarray = model.predict(xs)

    # Print Results
    print(pre_train_y)
    print()
    print(post_train_y)
