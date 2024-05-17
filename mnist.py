import numpy as np
from loguru import logger

from nn.activation import ReLU, Sigmoid
from nn.layer import Dense
from nn.loss import BinaryCrossEntropy
from nn.model import NeuralNetwork
from nn.optimizer import SGD, Adam, RMSprop

"""Download and unzip MNIST dataset"""

import asyncio
import gzip
import os
import shutil
from typing import Awaitable, Tuple

import numpy as np
import requests
from loguru import logger
from tqdm import tqdm


def _unzip_file(source_path: str):
    target_path = os.path.splitext(source_path)[0]

    logger.info(f"Unzipping file at {target_path}")

    with gzip.open(source_path, "rb") as f_in:
        with open(target_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


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
                for chunk in tqdm(res):
                    f.write(chunk)

            _unzip_file(file_path)
            return file_path
    return None


async def _download_url_async(url: str, data_dir: str) -> Awaitable:
    return await asyncio.to_thread(_download_url, url, data_dir)


def _read_file(
    file_path: str, buffer_size: int, discard_buffer_size: int
) -> np.ndarray:
    """Read a file and return its content as a numpy array"""

    logger.info(f"Reading file at path {file_path}")

    with open(file_path, "rb") as f:
        _ = f.read(discard_buffer_size)
        buffer = f.read(buffer_size)
    return np.frombuffer(buffer, dtype=np.uint8)


def _slice_data(
    images: np.ndarray, labels: np.ndarray, image_size: int
) -> Tuple[np.ndarray]:
    logger.info("Slicing data")

    zero_indices, *_ = np.where(np.squeeze(labels) == 0.0)
    one_indices, *_ = np.where(np.squeeze(labels) == 1.0)

    x = np.concatenate([images[zero_indices], images[one_indices]], axis=0)
    y = np.concatenate([labels[zero_indices], labels[one_indices]], axis=0)

    indices = np.random.permutation(y.shape[0])
    x = x[indices]
    y = y[indices]

    num_examples = x.shape[0]

    x = np.reshape(x, (num_examples, image_size * image_size))
    x = np.transpose(x) / 255.0
    y = np.transpose(y)
    return x, y


def _decode_data(
    image_path: str, label_path: str, num_images: int, image_size: int
) -> Tuple[np.ndarray]:
    """Read images and labels and return decoded numpy arrays"""

    logger.info(f"Decoding images at {image_path} and labels at {label_path}")

    buffer_size = image_size * image_size * num_images

    image_data = _read_file(image_path, buffer_size=buffer_size, discard_buffer_size=16)
    image_data = image_data.astype(np.float32)
    images = image_data.reshape(num_images, image_size, image_size)

    label_data = _read_file(label_path, buffer_size=buffer_size, discard_buffer_size=8)
    labels = label_data.reshape(num_images, 1)

    return images, labels


def _decode_and_slice_data(
    image_path: str, label_path: str, num_images: int, image_size: int
) -> Tuple[np.ndarray]:
    images, labels = _decode_data(image_path, label_path, num_images, image_size)
    return _slice_data(images, labels, image_size)


async def _download_dataset(data_dir):
    """Download dataset to data_dir"""

    base_url = "https://raw.githubusercontent.com/fgnt/mnist/master"
    files = (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )
    urls = [os.path.join(base_url, file) for file in files]

    os.makedirs(data_dir, exist_ok=True)

    await asyncio.gather(*[_download_url_async(url, data_dir) for url in urls])


def _create_dataset(data_dir: str) -> Tuple[np.ndarray]:
    num_test_images = 10000
    num_train_images = 60000
    image_size = 28

    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

    x_train, y_train = _decode_and_slice_data(
        train_images_path, train_labels_path, num_train_images, image_size
    )
    x_test, y_test = _decode_and_slice_data(
        test_images_path, test_labels_path, num_test_images, image_size
    )

    return x_train, y_train, x_test, y_test


def _download(data_dir: str):
    asyncio.run(_download_dataset(data_dir))


def mnist_load(data_dir: str) -> Tuple[np.ndarray]:
    _download(data_dir)
    return _create_dataset(data_dir)


def main():
    logger.info("Fetching dataset")

    x_train, y_train, x_test, y_test = mnist_load("data")

    logger.info("Creating model")

    model = NeuralNetwork(
        layers=(
            (Dense(64), ReLU()),
            (Dense(64), ReLU()),
            (Dense(1), Sigmoid()),
        ),
        loss=BinaryCrossEntropy(),
        optimizer=Adam(learning_rate=0.01),
        regularization_factor=2.0,
    )

    logger.info("Training model")

    model.fit(x_train, y_train, epochs=20, verbose=True)

    logger.info("Evaluating trained model")

    loss = model.evaluate(x_test, y_test)

    logger.info(f"Validation loss: {np.squeeze(loss):.4f}")

    preds = model.predict(x_test)
    preds = (preds >= 0.5).astype(int)

    logger.info(f"First 5 predictions: {preds[:, :5]}")
    logger.info(f"First 5 labels     : {y_test[:, :5]}")

    acc = np.squeeze(np.mean(preds == y_test))

    logger.info(f"Test set accuracy  : {acc:.4f}")


if __name__ == "__main__":
    main()
