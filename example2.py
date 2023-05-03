import numpy as np
from loguru import logger

import boston
from boston import _normalize_data, _standardize_data
from nn.activation import ReLU, Linear
from nn.layer import Dense
from nn.loss import MeanSquaredError, BinaryCrossEntropy, MeanAbsoluteError
from nn.model import NeuralNetwork
from nn.optimizer import Adam, SGD, RMSprop


def main():
    logger.info("Creating dataset")

    x_train, y_train, x_test, y_test = boston.load(preprocess_fn=_normalize_data)

    logger.info("Creating model")

    model = NeuralNetwork(
        layers=(
            (Dense(13), ReLU()),
            (Dense(128), ReLU()),
            (Dense(64), ReLU()),
            (Dense(1), Linear()),
        ),
        loss=MeanAbsoluteError(),
        optimizer=SGD(learning_rate=3.0),
        regularization_factor=0.01,
    )

    logger.info("Training model")

    model.fit(x_train, y_train, epochs=120, verbose=True)

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
