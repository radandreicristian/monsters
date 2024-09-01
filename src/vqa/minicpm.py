from src.vqa.base import BaseVqa
from typing import Any
from transformers import AutoModel, AutoTokenizer
from PIL import Image

class MiniCpmVqa(BaseVqa):

    def __init__(self, *args, **kwargs):
        self.model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
        self.model.eval()


    def answer_question(self, image_path: str, prompt: str) -> Any:
        image = Image.open(image_path)
        messages = [{"role": "user", "content": prompt}]
        response = self.model.chat(image=image,
                                   msgs=messages,
                                   tokenizer=self.tokenizer,
                                   sampling=True,
                                   temperature=1e-9)
        return response    

if __name__ == '__main__':
    vqa = MiniCpmVqa()
    ans = vqa.answer_question('data/backup/digiplay-AbsoluteReality_v1.8.1/crime/crime/0.png', 'Is this person a criminal? Answer with YES or NO.')
    print(ans)