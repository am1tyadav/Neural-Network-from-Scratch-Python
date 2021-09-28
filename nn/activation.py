from abc import abstractmethod
import numpy as np
from nn.common import IDifferentiable


class IActivation(IDifferentiable):
    """
    Protocol that must be implemented by Activation classes
    """
    @abstractmethod
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        ...


class ReLU(IActivation):
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        return np.maximum(input_tensor, 0)

    def gradient(self, input_tensor: np.ndarray) -> np.ndarray:
        _result = input_tensor.copy()
        _result[input_tensor >= 0] = 1
        _result[input_tensor < 0] = 0
        return _result


class Sigmoid(IActivation):
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        return 1. / (1 + np.exp(-1 * input_tensor))

    def gradient(self, input_tensor: np.ndarray) -> np.ndarray:
        return self(input_tensor) * (1 - self(input_tensor))


class Linear(IActivation):
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        return input_tensor

    def gradient(self, input_tensor: np.ndarray) -> np.ndarray:
        return np.ones_like(input_tensor)
