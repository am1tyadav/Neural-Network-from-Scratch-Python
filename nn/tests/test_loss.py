import numpy as np
import unittest
from nn.loss import BinaryCrossEntropy


class LossTest(unittest.TestCase):
    def test_gradient_descent(self):
        _num_examples = 100
        _num_features = 20
        _predictions = np.abs(np.random.randn(_num_features, _num_examples))
        _predictions = _predictions / np.max(_predictions)
        _labels = np.ones((_num_features, _num_examples)) - 0.01
        bce = BinaryCrossEntropy()
        _output_high = bce(_predictions, _labels)
        _output_low = bce(_labels, _labels)
        self.assertGreater(_output_high.mean(), _output_low.mean())
        self.assertGreater(0.1, float(_output_low))


if __name__ == "__main__":
    unittest.main()
