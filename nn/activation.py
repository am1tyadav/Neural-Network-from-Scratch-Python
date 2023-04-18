from abc import abstractmethod

import numpy as np

from nn.common import Differentiable


"""
1. The activations.py file defines a base abstract class called "Activation" and three concrete activation functions "ReLU", "Sigmoid", and "Linear".
2. All activation functions in this file implement the Differentiable interface, which requires the implementation of the gradient() method.
3. ReLU is a rectified linear unit activation function. It returns the maximum of the input tensor and 0.
4. The gradient of ReLU is implemented in the gradient() method. It returns a tensor with the same shape as the input tensor, where each element is either 1 or 0, depending on whether the corresponding element in the input tensor is greater than or equal to 0 or not.
5. Sigmoid is a sigmoidal activation function. It returns the result of the sigmoid function applied to the input tensor.
6. The gradient of Sigmoid is implemented in the gradient() method. It returns a tensor with the same shape as the input tensor, where each element is the derivative of the sigmoid function applied to the corresponding element in the input tensor.
7. Linear is a linear activation function. It returns the input tensor unchanged.
8. The gradient of Linear is implemented in the gradient() method. It returns a tensor with the same shape as the input tensor, where each element is 1.
"""


class Activation(Differentiable):
    """
    Protocol that must be implemented by Activation classes
    """

    @abstractmethod
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        ...


class ReLU(Activation):
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        return np.maximum(input_tensor, 0)

    def gradient(self, input_tensor: np.ndarray) -> np.ndarray:
        _result = input_tensor.copy()
        _result[input_tensor >= 0] = 1
        _result[input_tensor < 0] = 0
        return _result


class Sigmoid(Activation):
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        return 1.0 / (1 + np.exp(-1 * input_tensor))

    def gradient(self, input_tensor: np.ndarray) -> np.ndarray:
        return self(input_tensor) * (1 - self(input_tensor))


class Linear(Activation):
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        return input_tensor

    def gradient(self, input_tensor: np.ndarray) -> np.ndarray:
        return np.ones_like(input_tensor)
