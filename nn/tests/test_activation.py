import unittest

import numpy as np

import nn.activation


class ActivationTest(unittest.TestCase):
    def test_relu(self):
        _input = np.array([-1, 1])
        _output = np.array([0, 1])

        relu = nn.activation.ReLU()

        # Checking shapes
        self.assertEqual(relu(np.random.randn(10, 20)).shape, (10, 20))
        self.assertEqual(relu.gradient(np.random.randn(10, 20)).shape, (10, 20))
        # Checking values
        self.assertEqual(relu(_input).tolist(), _output.tolist())
        self.assertEqual(relu.gradient(_input).tolist(), _output.tolist())

    def test_sigmoid(self):
        _input = np.array([-100, 0, 100])

        sigmoid = nn.activation.Sigmoid()

        # Checking shapes
        self.assertEqual(sigmoid(np.random.randn(10, 20)).shape, (10, 20))
        self.assertEqual(sigmoid.gradient(np.random.randn(10, 20)).shape, (10, 20))
        # Checking values
        self.assertAlmostEqual(float(sigmoid(_input)[0]), 0.0)
        self.assertAlmostEqual(float(sigmoid(_input)[1]), 0.5)
        self.assertAlmostEqual(float(sigmoid(_input)[2]), 1.0)
        self.assertAlmostEqual(float(sigmoid.gradient(_input)[0]), 0.0)
        self.assertAlmostEqual(float(sigmoid.gradient(_input)[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
