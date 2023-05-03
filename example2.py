import numpy as np
from loguru import logger

import boston
from boston import _normalize_data, _orignal_data
from nn.activation import ReLU
from nn.layer import Dense
from nn.loss import MeanSquaredError, BinaryCrossEntropy, MeanAbsoluteError
from nn.model import NeuralNetwork
from nn.optimizer import Adam, SGD


def main():
    logger.info("Creating dataset")

    x_train, y_train, x_test, y_test = boston.load(preprocess_fn=_orignal_data)
    print(y_train.shape)
    print(x_train.shape)
    print(y_test.shape)
    print(x_test.shape)

    logger.info("Creating model")

    model = NeuralNetwork(
        layers=(
            (Dense(13), ReLU()),
            (Dense(18), ReLU()),
            (Dense(18), ReLU()),
            (Dense(1), ReLU()),
        ),
        loss=MeanAbsoluteError(),
        optimizer=Adam(learning_rate=0.03),
        regularization_factor=0.1,
    )

    logger.info("Training model")

    model.fit(x_train, y_train, epochs=20, verbose=True)

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
