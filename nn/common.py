from abc import ABC, abstractmethod
from typing import Any


class Differentiable(ABC):
    @abstractmethod
    def gradient(self, *args, **kwargs) -> Any: ...
