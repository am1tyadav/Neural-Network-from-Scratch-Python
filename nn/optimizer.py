import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        self.layer_number = 0

    @property
    def ln(self):
        return self.layer_number

    @ln.setter
    def ln(self, ln: int):
        self.layer_number = ln

    def update_weights(self, layer, grad_weights):

        if not isinstance(grad_weights, np.ndarray):
            raise TypeError("grad_weights should be a NumPy array")
        layer.weights -= self.learning_rate * grad_weights

    def update_bias(self, layer, grad_bias):

        if not isinstance(grad_bias, np.ndarray):
            raise TypeError("grad_bias should be a NumPy array")
        layer.bias -= self.learning_rate * grad_bias


class SGD(Optimizer):

    def __init__(self, learning_rate: float):
        super().__init__(learning_rate)

    def update_weights(self, layer, grad_weights):

        if not isinstance(grad_weights, np.ndarray):
            raise TypeError("grad_weights should be a NumPy array")
        layer.weights -= self.learning_rate * grad_weights

    def update_bias(self, layer, grad_bias):

        if not isinstance(grad_bias, np.ndarray):
            raise TypeError("grad_bias should be a NumPy array")
        layer.bias -= self.learning_rate * grad_bias


class Adam(Optimizer):

    def __init__(
        self,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}
        self.pw = 1

    def update_weights(self, layer, grad_weights: np.ndarray):

        if not isinstance(grad_weights, np.ndarray):
            raise TypeError("grad_weights should be a NumPy array")
        if not self.layer_number in self.m_w.keys():
            self.m_w[self.layer_number] = np.zeros_like(grad_weights)
            self.v_w[self.layer_number] = np.zeros_like(grad_weights)
        self.pw += 1
        self.m_w[self.layer_number] = (
            self.beta_1 * self.m_w[self.layer_number] + (1 - self.beta_1) * grad_weights
        )
        self.v_w[self.layer_number] = self.beta_2 * self.v_w[self.layer_number] + (
            1 - self.beta_2
        ) * (grad_weights**2)
        m_hat = self.m_w[self.layer_number] / (1 - self.beta_1**self.pw)
        v_hat = self.v_w[self.layer_number] / (1 - self.beta_2**self.pw)
        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def update_bias(self, layer, grad_bias: np.ndarray):

        if not isinstance(grad_bias, np.ndarray):
            raise TypeError("grad_bias should be a NumPy array")
        if not self.layer_number in self.m_b.keys():
            self.m_b[self.layer_number] = np.zeros_like(grad_bias)
            self.v_b[self.layer_number] = np.zeros_like(grad_bias)
        self.pw += 1
        self.m_b[self.layer_number] = (
            self.beta_1 * self.m_b[self.layer_number] + (1 - self.beta_1) * grad_bias
        )
        self.v_b[self.layer_number] = self.beta_2 * self.v_b[self.layer_number] + (
            1 - self.beta_2
        ) * (grad_bias**2)
        m_hat = self.m_b[self.layer_number] / (1 - self.beta_1**self.pw)
        v_hat = self.v_b[self.layer_number] / (1 - self.beta_2**self.pw)
        layer.bias -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        decay_rate: float = 0.9,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache_w = {}
        self.cache_b = {}

    def update_weights(self, layer, grad_weights: np.ndarray):

        if not self.layer_number in self.cache_w.keys():
            self.cache_w[self.layer_number] = np.zeros_like(grad_weights)
        self.cache_w[self.layer_number] = self.decay_rate * self.cache_w[
            self.layer_number
        ] + (1 - self.decay_rate) * (grad_weights**2)
        layer.weights -= (
            self.learning_rate
            * grad_weights
            / (np.sqrt(self.cache_w[self.layer_number]) + self.epsilon)
        )

    def update_bias(self, layer, grad_bias: np.ndarray):

        if not self.layer_number in self.cache_b.keys():
            self.cache_b[self.layer_number] = np.zeros_like(grad_bias)
        self.cache_b[self.layer_number] = self.decay_rate * self.cache_b[
            self.layer_number
        ] + (1 - self.decay_rate) * (grad_bias**2)
        layer.bias -= (
            self.learning_rate
            * grad_bias
            / (np.sqrt(self.cache_b[self.layer_number]) + self.epsilon)
        )
