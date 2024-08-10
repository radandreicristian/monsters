from src.llm.base import BaseLlmClient
from src.utils.prompts import add_attributes_to_template_prompt
from openai import AsyncOpenAI


class OpenAiLlmClient(BaseLlmClient):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.api_key = kwargs.get("OPENAI_API_KEY")
        self.model_name = kwargs.get("OPENAI_MODEL_NAME")

        self.openai_client = AsyncOpenAI(api_key=self.api_key)

        self.system_prompt = add_attributes_to_template_prompt

    def create_payload(self, prompt: str):
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system",
                 "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": prompt},
                    ],
                }
            ],
        }

    async def reply(self, prompt: str) -> str:
        payload = self.create_payload(prompt)
        response = await self.openai_client.chat.completions.create(**payload, temperature=1e-9)
        return response.choices[0].message.content
