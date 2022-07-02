import unittest

import numpy as np

import nn.layer


class LayerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._num_units = 10
        self._layer = nn.layer.Dense(units=self._num_units)

    def test_dense_shape(self):
        self.assertEqual(self._layer(np.random.randn(8, 5)).shape, (self._num_units, 5))


if __name__ == "__main__":
    unittest.main()
