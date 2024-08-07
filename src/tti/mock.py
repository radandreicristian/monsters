from typing import Any

import numpy as np

from PIL import Image

from src.tti.base import BaseTextToImage


class MockTextToImage(BaseTextToImage):

    def setup(self):
        pass

    def generate_image(self, prompt: str) -> Any:
        array = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        image = Image.fromarray(array)
        return image

    def store_image(self, image: Any, path: str) -> None:
        image.save(path)
