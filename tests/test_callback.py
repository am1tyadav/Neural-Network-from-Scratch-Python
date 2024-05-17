import os
import unittest

from nn.callback import CSVLogger


class CallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self._file_name = "test_csv_logger.csv"
        self._csv_logger = CSVLogger(self._file_name)

    def test_csv_logger(self):
        self._csv_logger.on_epoch_end(10, 0.1)

        with open(self._file_name, "r") as f:
            data = f.read().splitlines()

        self.assertEqual(data[0], "Epoch,Loss")
        self.assertEqual(data[1], "10,0.1")

    def tearDown(self) -> None:
        if os.path.isfile(self._file_name):
            os.remove(self._file_name)
