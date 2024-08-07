import abc
from typing import Any


class BaseTextToImage(abc.ABC):

    def setup(self):
        pass

    def generate_image(self, prompt: str) -> Any:
        pass

    def store_image(self, image: Any, path: str) -> None:
        pass
