from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from nn.activation import Activation
from nn.callback import Callback
from nn.layer import Layer
from nn.loss import Loss
from nn.optimizer import Optimizer


"""
1. The Model class is an abstract base class that defines the common interface for all models in the package. It contains several abstract methods that all models must implement, such as __call__, fit, predict, evaluate, backward_step, and update.
2. The NeuralNetwork class is a concrete implementation of the Model abstract base class. It represents a feedforward neural network that consists of a series of layers with associated activation functions.
3. The __init__ method of NeuralNetwork takes a list of tuples, where each tuple represents a layer and its associated activation function. The first item in each tuple must be a Layer object, and the second item must be an Activation object.
4. The __call__ method of NeuralNetwork implements the forward pass of the network. It takes an input tensor and passes it through each layer in the network, applying the associated activation function to the output of each layer.
5. The backward_step method of NeuralNetwork implements the backward pass of the network. It takes the true labels for a batch of examples and computes the gradients of the loss with respect to the weights and biases of each layer in the network. It then uses these gradients to update the weights and biases of each layer using the optimizer.
6. The fit method of NeuralNetwork trains the network on a given set of examples and labels. It takes the examples, labels, and the number of epochs to train for as input, and optionally takes a list of callbacks to be called after each epoch.
7. The predict method of NeuralNetwork takes a set of examples as input and returns the predicted labels for those examples. It does this by applying the forward pass of the network to the examples and thresholding the output at 0.5.
8. The evaluate method of NeuralNetwork takes a set of examples and their true labels as input, and returns the value of the loss function for those examples.
9. The update method of NeuralNetwork updates the weights and biases of each layer in the network using the optimizer.
"""


class Model(ABC):
    """Abstract base model class"""

    @property
    @abstractmethod
    def learning_rate(self):
        ...

    @learning_rate.setter
    @abstractmethod
    def learning_rate(self, value: float):
        ...

    @abstractmethod
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def fit(self, examples: np.ndarray, labels: np.ndarray, epochs: int):
        ...

    @abstractmethod
    def predict(self, examples: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def evaluate(self, examples: np.ndarray, labels: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward_step(self, labels: np.ndarray):
        ...

    @abstractmethod
    def update(self):
        ...


class NeuralNetwork(Model):
    """
    Todo - Optimizer needs to be separated from model implementation
    Instantiate with a list of of tuples. First item of each tuple must be a Layer
    and second item of each tuple must be an Activation applied to output of the layer
    """

    def __init__(
        self,
        layers: Tuple[Tuple[Layer, Activation]],
        loss: Loss,
        optimizer: Optimizer,
        regularization_factor: float = 0.0,
    ):
        self._layers = layers
        self._num_layers = len(layers)
        self._loss = loss
        self._optimizer = optimizer
        self._regularization_factor = regularization_factor
        self._input = None
        self._output = None
        self._num_examples = None

    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        :param input_tensor: (num_features, num_examples)
        :return: (num_units_of_final_layer, num_examples)
        """
        if self._num_examples is None:
            self._num_examples = input_tensor.shape[-1]

        output = input_tensor

        for layer, activation in self._layers:
            output = layer(output)
            output = activation(output)

        self._output = output
        return self._output

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value


    def backward_step(self, labels: np.ndarray):
        da = self._loss.gradient(self._output, labels)

        for index in reversed(range(0, self._num_layers)):
            layer, activation = self._layers[index]

            if index == 0:
                prev_layer_output = self._input
            else:
                prev_layer, prev_activation = self._layers[index - 1]
                prev_layer_output = prev_activation(prev_layer.output)

            dz = np.multiply(da, activation.gradient(layer.output))
            layer.grad_weights = (
                np.dot(dz, np.transpose(prev_layer_output)) / self._num_examples
            )
            layer.grad_weights = (
                layer.grad_weights
                + (self._regularization_factor / self._num_examples) * layer.get_weights()
            )
            layer.grad_bias = np.mean(dz, axis=1, keepdims=True)
            da = np.dot(np.transpose(layer.get_weights()), dz)
            
            self._optimizer.update_weights(layer, layer.grad_weights)
            self._optimizer.update_bias(layer, layer.grad_bias)


    def fit(
        self,
        examples: np.ndarray,
        labels: np.ndarray,
        epochs: int,
        verbose: bool = False,
        callbacks: Tuple[Callback] = (),
    ):
        for epoch in range(1, epochs + 1):
            self._input = examples
            _ = self(self._input)
            loss = self._loss(self._output, labels)
            self.backward_step(labels)
            self.update()

            for callback in callbacks:
                loss_scalar = float(np.squeeze(loss))
                callback.on_epoch_end(epoch, loss_scalar)

            if verbose:
                print(f"Epoch: {epoch:03d}, Loss {loss:0.4f}")

    def predict(self, examples: np.ndarray) -> np.ndarray:
        outputs = self(examples)
        return (outputs > 0.5).astype("uint8")

    def evaluate(self, examples: np.ndarray, labels: np.ndarray) -> np.ndarray:
        _ = self(examples)
        return self._loss(self._output, labels)

    def update(self):
        for layer, _ in self._layers:
            layer.update(self._optimizer)
