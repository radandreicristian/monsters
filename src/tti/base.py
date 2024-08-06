import abc


class BaseTextToImage(abc.ABC):

    def setup(self):
        pass

    def generate_image(self, prompt: str, image_path: str) -> None:
        pass
