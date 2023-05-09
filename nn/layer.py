from abc import ABC, abstractmethod
import numpy as np
from nn.optimizer import Optimizer

class Layer(ABC):
    """Abstract base class for neural network layers.

    Attributes:
        output (numpy.ndarray): The output tensor of the layer.

    Methods:
        __call__(input_tensor: numpy.ndarray) -> numpy.ndarray:
            Applies the layer to the input tensor and returns the output.
        build(input_tensor: numpy.ndarray):
            Builds the layer and initializes its weights and biases.
        update(optimizer: Optimizer):
            Updates the layer's weights and biases using the specified optimizer.
    """
    @property
    @abstractmethod
    def output(self):
        ...

    @abstractmethod
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def build(self, input_tensor: np.ndarray):
        ...

    @abstractmethod
    def update(self, optimizer: Optimizer):
        ...


class Dense(Layer):
    def __init__(self, units: int):
        self._units = units
        self._input_units = None
        self._weights = None
        self._bias = None
        self._output = None
        self._dw = None
        self._db = None

    @property
    def weights(self):
        return self._weights
        
    @weights.setter
    def weights(self, weights: np.ndarray):
        self._weights = weights

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias: np.ndarray):
        self._bias = bias

    @property
    def grad_weights(self):
        return self._dw

    @grad_weights.setter
    def grad_weights(self, gradients: np.ndarray):
        self._dw = gradients

    @property
    def grad_bias(self):
        return self._db

    @grad_bias.setter
    def grad_bias(self, gradients: np.ndarray):
        self._db = gradients

    @property
    def output(self):
        return self._output

    def build(self, input_tensor: np.ndarray):
        self._input_units = input_tensor.shape[0]
        self._weights = np.random.randn(self._units, self._input_units) * np.sqrt(
            2.0 / self._input_units
        )
        self._bias = np.zeros((self._units, 1))

    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        # _input_shape = input_tensor.shape

        if self._weights is None:
            self.build(input_tensor)

        self._output = np.dot(self._weights, input_tensor) + self._bias
        return self._output

    def update(self, optimizer: Optimizer):
        optimizer.update_weights(self, self.grad_weights)
        optimizer.update_bias(self, self.grad_bias)