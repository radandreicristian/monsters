from abc import ABC, abstractmethod


class BaseImageAttributeExtractor(ABC):

    @abstractmethod
    def extract_attributes(self, images_root: str) -> list[str]:
        pass
