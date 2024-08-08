from abc import ABC, abstractmethod


class BaseImageAttributeExtractor(ABC):

    @abstractmethod
    async def extract_attributes(self, images_root: str) -> list[str]:
        pass
