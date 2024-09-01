from abc import ABC, abstractmethod
from typing import Any


class BaseTextToImage(ABC):

    def setup(self):
        pass

    @abstractmethod
    def generate_images(self, prompt: str, n_images: int) -> Any:
        pass

    @abstractmethod
    def store_image(self, image: Any, path: str) -> None:
        pass
