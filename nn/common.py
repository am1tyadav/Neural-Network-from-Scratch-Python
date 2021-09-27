from typing import Any
from abc import ABC, abstractmethod


class IDifferentiable(ABC):
    @abstractmethod
    def gradient(self, *args, **kwargs) -> Any:
        ...
