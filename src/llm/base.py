from abc import ABC, abstractmethod
from typing import Any


class BaseLlmClient(ABC):

    async def setup(self):
        pass

    @abstractmethod
    def reply(self, prompt: str) -> Any:
        pass

