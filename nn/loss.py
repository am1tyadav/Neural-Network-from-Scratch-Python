from abc import abstractmethod
import numpy as np
from nn.common import IDifferentiable


class ILoss(IDifferentiable):
    """
    This protocol must be implemented by Loss classes
    """
    @abstractmethod
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        ...


class BinaryCrossEntropy(ILoss):
    def __call__(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        loss = np.mean(np.multiply(labels, np.log(predictions)) + np.multiply(1 - labels, np.log(1 - predictions)))
        return -1 * loss

    def gradient(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return -1 * (np.divide(labels, predictions) - np.divide(1 - labels, 1 - predictions))
