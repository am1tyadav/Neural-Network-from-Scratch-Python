from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from nn.common import Differentiable


class Loss(Differentiable):
    """
    This abstract class must be implemented by concrete Loss classes
    """

    @abstractmethod
    def __call__(self, predictions: NDArray, labels: NDArray) -> NDArray: ...


class BinaryCrossEntropy(Loss):
    def __call__(self, predictions: NDArray, labels: NDArray) -> NDArray:
        epsilon = 1e-7  # Small epsilon value to avoid division by zero
        loss = np.multiply(labels, np.log(predictions + epsilon)) + np.multiply(
            1 - labels, np.log(1 - predictions + epsilon)
        )
        return -1 * np.mean(loss)

    def gradient(self, predictions: NDArray, labels: NDArray) -> NDArray:
        epsilon = 1e-7  # Small epsilon value to avoid division by zero
        return -1 * (
            np.divide(labels, predictions + epsilon)
            - np.divide(1 - labels, 1 - predictions + epsilon)
        )


class MeanSquaredError(Loss):
    def __call__(self, predictions: NDArray, labels: NDArray) -> NDArray:
        return np.mean(np.square(predictions - labels))

    def gradient(self, predictions: NDArray, labels: NDArray) -> NDArray:
        return 2 * (predictions - labels) / labels.size


class MeanAbsoluteError(Loss):
    def __call__(self, predictions: NDArray, labels: NDArray) -> NDArray:
        return np.mean(np.abs(predictions - labels))

    def gradient(self, predictions: NDArray, labels: NDArray) -> NDArray:
        return np.sign(predictions - labels) / labels.size


class HuberLoss(Loss):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(self, predictions: NDArray, labels: NDArray) -> NDArray:
        diff = np.abs(predictions - labels)
        quadratic = np.minimum(diff, self.delta)
        linear = diff - quadratic
        return np.mean(0.5 * np.square(quadratic) + self.delta * linear)

    def gradient(self, predictions: NDArray, labels: NDArray) -> NDArray:
        diff = predictions - labels
        is_small_error = np.abs(diff) <= self.delta
        gradient = np.zeros_like(diff)
        gradient[is_small_error] = diff[is_small_error]
        gradient[~is_small_error] = self.delta * np.sign(diff[~is_small_error])
        return gradient / labels.size
