import torch
from src.tti.base import BaseTextToImage

from typing import Any
from src.logger.utils import get_logger
from src.utils.prompts import sdxl_negative_prompt, tti_positive_prompt

logger = get_logger()
from diffusers import AutoPipelineForText2Image
import torch

class SdxlTextToImage(BaseTextToImage):

    def __init__(self, 
                 *args, **kwargs):
        
        self.generator = torch.Generator(device="cuda")
        self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def generate_images(self, prompt: str, n_images: int) -> Any:
        images = []
        for i in range(n_images):
            self.generator.manual_seed(i)
            image = self.pipe(prompt=prompt+tti_positive_prompt, 
                              negative_prompt=sdxl_negative_prompt,
                              width=512,
                              height=512, 
                              generator=self.generator, num_inference_steps=4, guidance_scale=0.0).images[0]
            images.append(image)
        return images
 
    def store_image(self, image: Any, path: str) -> None:
        image.save(path)

if __name__ == "__main__":
    from src.utils.deterministic import make_deterministic
    make_deterministic()

    tti = SdxlTextToImage()
    
    # Run 1
    r1_images = tti.generate_images("photograph of a criminal, color", n_images = 2)
    for i, image in enumerate(r1_images):
        tti.store_image(image, f"sdxl_r1_{i}.png")

    r2_images = tti.generate_images("photograph of a criminal, color", n_images = 2)
    for i, image in enumerate(r2_images):
        tti.store_image(image, f"sdxl_r2_{i}.png")