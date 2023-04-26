import numpy as np
from loguru import logger

import mnist
from nn.activation import ReLU, Sigmoid
from nn.layer import Dense
from nn.loss import BinaryCrossEntropy
from nn.model import NeuralNetwork
from nn.optimizer import SGD, Adam, RMSprop


def main():
    logger.info("Fetching dataset")

    x_train, y_train, x_test, y_test = mnist.load("data")

    logger.info("Creating model")

    model = NeuralNetwork(
        layers=(
            (Dense(128), ReLU()),
            (Dense(128), ReLU()),
            (Dense(128), ReLU()),
            (Dense(1), Sigmoid()),
        ),
        loss=BinaryCrossEntropy(),
        optimizer=SGD(learning_rate=0.03),
        regularization_factor=2.0,
    )

    logger.info("Training model")

    model.fit(x_train, y_train, epochs=40, verbose=True)

    logger.info("Evaluating trained model")

    loss = model.evaluate(x_test, y_test)

    logger.info(f"Validation loss: {np.squeeze(loss):.4f}")

    preds = model.predict(x_test)

    logger.info(f"First 5 predictions: {preds[:, :5]}")
    logger.info(f"First 5 labels     : {y_test[:, :5]}")

    acc = np.squeeze(np.mean(preds == y_test))

    logger.info(f"Test set accuracy  : {acc:.4f}")


if __name__ == "__main__":
    main()
