import asyncio
import os
import shutil
from typing import Awaitable, Tuple

import numpy as np
import requests
from loguru import logger


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


async def _download_url_async(url: str, data_dir: str) -> Awaitable:
    return await asyncio.to_thread(_download_url, url, data_dir)


def _read_file(file_path: str) -> np.ndarray:
    """Read a file and return its content as a numpy array"""

    logger.info(f"Reading file at path {file_path}")

    with open(file_path, "r") as f:
        data = f.read()

    return np.array([list(map(float, line.split())) for line in data.split("\n") if line.strip()])


def _decode_data(data_path: str) -> np.ndarray:
    """Read data and return decoded numpy array"""

    logger.info(f"Decoding data at {data_path}")

    return _read_file(data_path)


def _standardize_data(x: np.ndarray) -> np.ndarray:
    """Standardize data by subtracting mean and dividing by standard deviation"""

    logger.info("Standardizing data")

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    return (x - x_mean) / x_std


def _normalize_data(x: np.ndarray) -> np.ndarray:
    """Normalize data by dividing by max value"""

    logger.info("Normalizing data")

    return x / np.max(x, axis=0)


def _decode_and_preprocess_data(
    data_path: str, target_path: str, preprocess_fn
) -> Tuple[np.ndarray]:
    data = _decode_data(data_path)
    target = _decode_data(target_path)

    x = preprocess_fn(data)
    y = target

    return x, y


def train_test_split(x, y, test_size=0.3, random_state=None):
    """Split the data into train and test sets"""

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = x.shape[0]
    test_size = int(test_size * n_samples)
    shuffled_indices = np.random.permutation(n_samples)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, x_test, y_train, y_test


async def _download_dataset(data_dir):
    """Download dataset to data_dir"""

    urls = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names",
    )

    os.makedirs(data_dir, exist_ok=True)

    await asyncio.gather(*[_download_url_async(url, data_dir) for url in urls])


def load(preprocess_fn=_standardize_data, test_size=0.3):
    """Load the Boston Housing dataset and return a train-test split"""
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    if not os.path.isdir(data_dir):
        asyncio.run(_download_dataset(data_dir))

    data_path = os.path.join(data_dir, "housing.data")
    target_path = os.path.join(data_dir, "housing.names")

    x, y = _decode_and_preprocess_data(data_path, target_path, preprocess_fn)

    return train_test_split(x, y, test_size=0.3, random_state=42) # Split the data into training and testing sets with a 30/70 split ratio.

