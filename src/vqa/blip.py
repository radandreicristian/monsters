from src.vqa.base import BaseVqa
from typing import Any
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

class BlipVqa(BaseVqa):

    def __init__(self, *args, **kwargs):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")


    def answer_question(self, image_path: str, prompt: str) -> Any:
        image = Image.open(image_path)
        encoding = self.processor(image, prompt, return_tensors="pt")
        output = self.model.generate(**encoding)
        return self.processor.decode(output[0], skip_special_tokens=True)
    

if __name__ == '__main__':
    vqa = BlipVqa()
    ans = vqa.answer_question('data/backup/digiplay-AbsoluteReality_v1.8.1/crime/crime/0.png', 'Is this person a criminal? Answer with YES or NO.')
    print(ans)