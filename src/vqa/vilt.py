from src.vqa.base import BaseVqa
from typing import Any
from transformers import pipeline
from PIL import Image

class ViltVqa(BaseVqa):

    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get("model_name")
        self.pipe = pipeline("visual-question-answering", model=self.model_name)

    def answer_question(self, image_path: str, prompt: str) -> Any:
        image = Image.open(image_path)
        return self.pipe(image=image, question=prompt)
