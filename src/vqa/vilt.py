from src.vqa.base import BaseVqa
from typing import Any
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

class ViltVqa(BaseVqa):

    def __init__(self, *args, **kwargs):
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


    def answer_question(self, image_path: str, prompt: str) -> Any:
        image = Image.open(image_path)
        encoding = self.processor(image, prompt, return_tensors="pt")
        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return self.model.config.id2label[idx]
    

if __name__ == '__main__':
    vqa = ViltVqa()
    ans = vqa.answer_question('data/backup/digiplay-AbsoluteReality_v1.8.1/crime/crime/0.png', 'Is this person a criminal? Answer with YES or NO.')
    print(ans)