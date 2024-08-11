import abc
from typing import Any


class BaseVqa(abc.ABC):

    def answer_question(self, image: Any, prompt: str) -> Any:
        pass
