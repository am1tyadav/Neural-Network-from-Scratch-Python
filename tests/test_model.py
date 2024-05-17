import unittest
import numpy as np
from nn.activation import ReLU, Sigmoid
from nn.layer import Dense
from nn.loss import BinaryCrossEntropy
from nn.optimizer import SGD
from nn.model import NeuralNetwork


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        layers = [
            (Dense(units=128), ReLU()),
            (Dense(units=128), ReLU()),
            (Dense(units=10), Sigmoid()),
        ]
        loss = BinaryCrossEntropy()
        optimizer = SGD(learning_rate=0.01)
        self._NeuralNetwork = NeuralNetwork(
            layers=layers, loss=loss, optimizer=optimizer
        )

    def test_NeuralNetwork(self):
        _input = np.zeros((784, 16))
        _output = self._NeuralNetwork(_input)

        self.assertEqual(_output.shape, (10, 16))
        self.assertAlmostEqual(float(_output[0][0]), 0.5)


if __name__ == "__main__":
    unittest.main()
