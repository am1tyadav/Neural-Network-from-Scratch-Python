import asyncio
import os

import numpy as np
from numpy.typing import NDArray
import requests
from loguru import logger

from nn.activation import ReLU, Linear
from nn.layer import Dense
from nn.loss import MeanAbsoluteError
from nn.model import NeuralNetwork
from nn.optimizer import RMSprop


def _download_url(url: str, data_dir: str) -> str:
    """Download file at url to data_dir"""

    file_name = os.path.basename(url)
    file_path = os.path.join(data_dir, file_name)

    if os.path.isfile(file_path):
        logger.warning(f"{file_path} already exists")
    else:
        logger.info(f"Downloading file from {url}")
        res = requests.get(url=url, stream=True)

        if res.status_code == 200:
            logger.info("Got successful response, downloading..")

            with open(file_path, "wb") as f:
                f.write(res.content)

            return file_path
    return None


def _read_file(file_path: str) -> NDArray:
    """Read a file and return its content as a numpy array"""

    logger.info(f"Reading file at path {file_path}")

    with open(file_path, "r") as f:
        data = f.read()

    return np.array(
        [list(map(float, line.split())) for line in data.split("\n") if line.strip()]
    )


def _decode_data(data_path: str) -> NDArray:
    """Read data and return decoded numpy array"""

    logger.info(f"Decoding data at {data_path}")

    return _read_file(data_path)


def _orignal_data(x: NDArray) -> NDArray:
    """Return the orignal data as it is"""
    return x


def _normalize_data(x: NDArray) -> NDArray:
    """Normalize data by dividing by max value"""

    logger.info("Normalizing data")

    return x / np.max(x, axis=0)


def train_test_split(x, y, test_size=0.3, random_state=None):
    """Split the data into train and test sets"""

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = x.shape[0]
    test_size = int(test_size * n_samples)
    shuffled_indices = np.random.permutation(n_samples)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    x_train = x[train_indices].T
    y_train = y[train_indices].reshape(1, -1)
    x_test = x[test_indices].T
    y_test = y[test_indices].reshape(1, -1)

    return x_train, y_train, x_test, y_test


async def _download_dataset(data_dir):
    """Download dataset to data_dir"""

    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    )

    os.makedirs(data_dir, exist_ok=True)

    await asyncio.to_thread(_download_url, url, data_dir)


def boston_load(preprocess_fn=_orignal_data, test_size=0.3):
    """Load the Boston Housing dataset and return a train-test split"""

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    if not os.path.isdir(data_dir) or not os.path.isfile(
        os.path.join(data_dir, "housing.data")
    ):
        asyncio.run(_download_dataset(data_dir))

    data_path = os.path.join(data_dir, "housing.data")

    x = _decode_data(data_path)
    y = x[:, -1]  # The last column in the dataset is the target variable

    x = x[:, :-1]  # Remove the target variable from the input features

    x = preprocess_fn(x)
    y = preprocess_fn(y)

    return train_test_split(
        x, y, test_size=test_size, random_state=42
    )  # Split the data into training and testing sets with a 30/70 split ratio.


def r2_score(preds: NDArray, y_test: NDArray) -> float:
    mean_y = np.mean(y_test)
    ss_total = np.sum((y_test - mean_y) ** 2)
    ss_residual = np.sum((y_test - preds) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2


def main():
    logger.info("Creating dataset")

    x_train, y_train, x_test, y_test = boston_load(preprocess_fn=_normalize_data)

    logger.info("Creating model")

    model = NeuralNetwork(
        layers=(
            (Dense(13), ReLU()),
            (Dense(128), ReLU()),
            (Dense(64), ReLU()),
            (Dense(1), Linear()),
        ),
        loss=MeanAbsoluteError(),
        optimizer=RMSprop(learning_rate=0.0003),
        regularization_factor=0.001,
    )

    logger.info("Training model")

    model.fit(x_train, y_train, epochs=24000, verbose=True, log_interval=2000)

    logger.info("Evaluating trained model")

    loss = model.evaluate(x_test, y_test)

    logger.info(f"Validation loss: {np.squeeze(loss):.4f}")

    preds = model.predict(x_test)

    logger.info(f"First 5 predictions: {preds[:, :5]}")
    logger.info(f"First 5 labels     : {y_test[:, :5]}")

    acc = r2_score(preds, y_test)

    logger.info(f"Test set accuracy  : {acc:.4f}")


if __name__ == "__main__":
    main()
