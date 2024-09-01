from abc import abstractmethod, ABC
from typing import Any


class BaseVqa(ABC):

    @abstractmethod
    def answer_question(self, image_path: Any, prompt: str) -> Any:
        pass
