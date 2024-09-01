from typing import Any

import numpy as np

from PIL import Image

from src.tti.base import BaseTextToImage


class MockTextToImage(BaseTextToImage):

    def setup(self):
        pass

    def generate_images(self, prompt: str, n_images: int) -> Any:
        images = []
        for _ in range(n_images):
            array = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            images.append(Image.fromarray(array))
        return images

    def store_image(self, image: Any, path: str) -> None:
        image.save(path)
