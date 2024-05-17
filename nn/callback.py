import os
from abc import ABC, abstractmethod


class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch: int, loss: float): ...


class CSVLogger(Callback):
    def __init__(self, file_path: str, overwrite: bool = False):
        self._file_path = file_path
        if os.path.isfile(file_path) and not overwrite:
            raise FileExistsError(f"Log file already exists at {file_path}")
        else:
            with open(self._file_path, "w") as f:
                f.write("Epoch,Loss\n")

    def on_epoch_end(self, epoch: int, loss: float):
        with open(self._file_path, "a") as f:
            f.write(f"{epoch},{loss}\n")
