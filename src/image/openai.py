import ast
import logging
import os
from openai import AsyncOpenAI

from src.image.base import BaseImageAttributeExtractor
from src.utils.base64 import encode_image

logger = logging.getLogger()


class OpenAiImageAttributeExtractor(BaseImageAttributeExtractor):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.api_key = kwargs.get("OPENAI_API_KEY")
        self.model_name = kwargs.get("OPENAI_MODEL_NAME")

        self.openai_client = AsyncOpenAI(api_key=self.api_key)

        self.system_prompt = kwargs.get("OPENAI_IMAGE_FEATURE_EXTRACTION_SYSTEM_PROMPT")
        self.user_prompt = kwargs.get("OPENAI_IMAGE_FEATURE_EXTRACTION_USER_PROMPT")

    @staticmethod
    def create_image_payload(images_root: str):
        image_paths = [os.path.join(images_root, image_name) for image_name in os.listdir(images_root)]
        image_messages = [{
            "role": "user",
            "content": [
                {
                    "image_url": {
                        "url": encode_image(image_path),
                        "detail": "auto"
                    },
                    "type": "image_url"
                }
            ]
        } for image_path in image_paths]
        return image_messages

    def create_payload(self, images_root: str) -> dict:
        image_messages = self.create_image_payload(images_root=images_root)
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system",
                 "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": self.user_prompt},
                    ],
                },
                *image_messages
            ],
        }

    async def extract_attributes(self, images_root: str) -> list[str]:
        payload = self.create_payload(images_root=images_root)
        response = await self.openai_client.chat.completions.create(**payload, temperature=1e-9)
        content = response.choices[0].message.content
        try:
            parsed_response = ast.literal_eval(content)
            return parsed_response
        except SyntaxError:
            logger.error(f"Could not parse response {response}", exc_info=True)
            return []
