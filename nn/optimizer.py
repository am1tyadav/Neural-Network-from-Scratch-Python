import numpy as np

"""
1. The Optimizer class is an abstract base class that defines the interface for all optimizers.
2. It has two abstract methods update_weights and update_bias that are implemented by its subclasses.
3. The SGD class is a subclass of Optimizer that implements the stochastic gradient descent optimizer.
4. Its constructor takes a learning_rate parameter and calls the constructor of its superclass Optimizer.
5. The update_weights method of SGD updates the weights of a layer by subtracting the product of the learning rate and the gradient of the weights.
6. The update_bias method of SGD updates the bias of a layer by subtracting the product of the learning rate and the gradient of the bias.
7. The Adam optimizer is an extension of the stochastic gradient descent optimizer that uses adaptive learning rates for each weight parameter. It also includes momentum terms and a bias correction to the adaptive learning rate.
8. The RMSprop optimizer uses a moving average of the squared gradient to adapt the learning rate for each weight parameter.
"""

class Optimizer:

    """Base optimizer class for neural network optimization.
    
    Parameters:
    learning_rate (float): The learning rate to be used for optimization.
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        
    def update_weights(self, layer, grad_weights):
        """Update the weights of the given layer using the gradient of weights.
        
        Parameters:
        layer: A layer object whose weights need to be updated.
        grad_weights (np.ndarray): The gradient of weights computed during backpropagation.
        
        Raises:
        TypeError: If grad_weights is not a NumPy array.
        """
        if not isinstance(grad_weights, np.ndarray):
            raise TypeError("grad_weights should be a NumPy array")
        layer.weights -= self.learning_rate * grad_weights
        
    def update_bias(self, layer, grad_bias):
        """Update the bias of the given layer using the gradient of bias.
        
        Parameters:
        layer: A layer object whose bias needs to be updated.
        grad_bias (np.ndarray): The gradient of bias computed during backpropagation.
        
        Raises:
        TypeError: If grad_bias is not a NumPy array.
        """
        if not isinstance(grad_bias, np.ndarray):
            raise TypeError("grad_bias should be a NumPy array")
        layer.bias -= self.learning_rate * grad_bias

class SGD(Optimizer):

    """Stochastic Gradient Descent optimizer class for neural network optimization.
    
    Parameters:
    learning_rate (float): The learning rate to be used for optimization.
    """

    def __init__(self, learning_rate: float):
        super().__init__(learning_rate)
        
    def update_weights(self, layer, grad_weights):
        """Update the weights of the given layer using the gradient of weights.
        
        Parameters:
        layer: A layer object whose weights need to be updated.
        grad_weights (np.ndarray): The gradient of weights computed during backpropagation.
        
        Raises:
        TypeError: If grad_weights is not a NumPy array.
        """
        if not isinstance(grad_weights, np.ndarray):
            raise TypeError("grad_weights should be a NumPy array")
        layer.weights -= self.learning_rate * grad_weights
        
    def update_bias(self, layer, grad_bias):
        """Update the bias of the given layer using the gradient of bias.
        
        Parameters:
        layer: A layer object whose bias needs to be updated.
        grad_bias (np.ndarray): The gradient of bias computed during backpropagation.
        
        Raises:
        TypeError: If grad_bias is not a NumPy array.
        """
        if not isinstance(grad_bias, np.ndarray):
            raise TypeError("grad_bias should be a NumPy array")
        layer.bias -= self.learning_rate * grad_bias


class Adam(Optimizer):

    """Adam optimizer class for neural network optimization.
    
    Parameters:
    learning_rate (float): The learning rate to be used for optimization.
    beta_1 (float): The exponential decay rate for the first moment estimates.
    beta_2 (float): The exponential decay rate for the second moment estimates.
    epsilon (float): A small value added to the denominator to avoid dividing by zero.
    """

    def __init__(self, learning_rate: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def update_weights(self, layer, grad_weights: np.ndarray):
        """
        Updates the weights of a layer using Adam optimizer.

        Args:
            layer: The layer object to update.
            grad_weights: A NumPy array of gradients of the weights of the layer.

        Raises:
            TypeError: If grad_weights is not a NumPy array.

        """
        if self.m is None:
            self.m = np.zeros_like(grad_weights)
            self.v = np.zeros_like(grad_weights)
        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_weights
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_weights ** 2)
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
    def update_bias(self, layer, grad_bias: np.ndarray):
        """
        Updates the bias of a layer using Adam optimizer.

        Args:
            layer: The layer object to update.
            grad_bias: A NumPy array of gradients of the bias of the layer.

        Raises:
            TypeError: If grad_bias is not a NumPy array.

        """
        if self.m is None:
            self.m = np.zeros_like(grad_bias)
            self.v = np.zeros_like(grad_bias)
        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_bias
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_bias ** 2)
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        layer.bias -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    def __init__(self, learning_rate: float = 0.001, decay_rate: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None
        
    def update_weights(self, layer, grad_weights: np.ndarray):
        """
        Updates the weights of a layer using RMSprop optimizer.

        Args:
            layer: The layer object to update.
            grad_weights: A NumPy array of gradients of the weights of the layer.

        Raises:
            TypeError: If grad_weights is not a NumPy array.

        """
        if self.cache is None:
            self.cache = np.zeros_like(grad_weights)
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * (grad_weights ** 2)
        layer.weights -= self.learning_rate * grad_weights / (np.sqrt(self.cache) + self.epsilon)
        
    def update_bias(self, layer, grad_bias: np.ndarray):
        """
        Updates the bias of a layer using RMSprop optimizer.

        Args:
            layer (Layer): The layer whose bias should be updated.
            grad_bias (np.ndarray): The gradient of the loss with respect to the bias.
        Raises:
            TypeError: If grad_bias is not a NumPy array.

        Returns:
            None. The bias of the layer is updated in-place.
        """
        if self.cache is None:
            self.cache = np.zeros_like(grad_bias)
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * (grad_bias ** 2)
        layer.bias -= self.learning_rate * grad_bias / (np.sqrt(self.cache) + self.epsilon)
