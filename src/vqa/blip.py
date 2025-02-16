from src.vqa.base import BaseVqa
from typing import Any
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

class BlipVqa(BaseVqa):

    def __init__(self, *args, **kwargs):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")  # type: ignore # noqa


    def answer_question(self, image_path: str, prompt: str) -> Any:
        image = Image.open(image_path)
        encoding = self.processor(image, prompt, return_tensors="pt").to("cuda")  # type: ignore # noqa
        output = self.model.generate(**encoding)  # type: ignore # noqa
        return self.processor.decode(output[0], skip_special_tokens=True)  # type: ignore # noqa
    

if __name__ == '__main__':
    vqa = BlipVqa()
    for i in range(4):
        for j in range(5):
            ans = vqa.answer_question(f'data/biased_images/stabilityai-stable-diffusion-3/drug_trafficking/direct/p_{i}_seed_{j}.png', 'Is this person a drug trafficker? Answer with YES or NO.')
            print(f"p_{i}_seed_{j}.png: {ans}")