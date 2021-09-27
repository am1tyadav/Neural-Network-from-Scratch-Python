import numpy as np
import unittest
from nn.model import MLP
from nn.layer import Dense
from nn.activation import ReLU, Sigmoid
from nn.loss import BinaryCrossEntropy


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        layers = [
            (Dense(units=128), ReLU()),
            (Dense(units=128), ReLU()),
            (Dense(units=10), Sigmoid())
        ]
        loss = BinaryCrossEntropy()
        self._mlp = MLP(layers=layers, loss=loss, lr=0.1)

    def test_mlp(self):
        _input = np.zeros((784, 16))
        _output = self._mlp(_input)

        self.assertEqual(_output.shape, (10, 16))
        self.assertAlmostEqual(float(_output[0][0]), 0.5)


if __name__ == '__main__':
    unittest.main()
