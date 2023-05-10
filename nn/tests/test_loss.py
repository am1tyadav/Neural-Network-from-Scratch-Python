import unittest
import numpy as np
from nn.loss import BinaryCrossEntropy, MeanSquaredError, MeanAbsoluteError, HuberLoss


class LossTest(unittest.TestCase):
    def test_gradient_descent(self):
        _num_examples = 100
        _num_features = 20
        _predictions = np.abs(np.random.randn(_num_features, _num_examples))
        _predictions = _predictions / np.max(_predictions)
        _labels = np.ones((_num_features, _num_examples)) - 0.01

        bce = BinaryCrossEntropy()
        mse = MeanSquaredError()
        mae = MeanAbsoluteError()
        huber = HuberLoss()

        _output_high_bce = bce(_predictions, _labels)
        _output_low_bce = bce(_labels, _labels)
        self.assertGreater(_output_high_bce.mean(), _output_low_bce.mean())
        self.assertGreater(0.1, float(_output_low_bce))

        _output_high_mse = mse(_predictions, _labels)
        _output_low_mse = mse(_labels, _labels)
        self.assertGreater(_output_high_mse.mean(), _output_low_mse.mean())
        self.assertGreater(0.1, float(_output_low_mse))

        _output_high_mae = mae(_predictions, _labels)
        _output_low_mae = mae(_labels, _labels)
        self.assertGreater(_output_high_mae.mean(), _output_low_mae.mean())
        self.assertGreater(0.1, float(_output_low_mae))

        _output_high_huber = huber(_predictions, _labels)
        _output_low_huber = huber(_labels, _labels)
        self.assertGreater(_output_high_huber.mean(), _output_low_huber.mean())
        self.assertGreater(0.1, float(_output_low_huber))


if __name__ == "__main__":
    unittest.main()
