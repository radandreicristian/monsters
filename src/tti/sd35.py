import torch
from src.tti.base import BaseTextToImage

from typing import Any
from src.logger.utils import get_logger
from src.utils.prompts import sdxl_negative_prompt, tti_positive_prompt

logger = get_logger()
from diffusers import DiffusionPipeline
import torch



class Sd35TextToImage(BaseTextToImage):

    def __init__(self, 
                 *args, **kwargs):
        
        self.generator = torch.Generator(device="cuda")

        p = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", device_map="balanced", max_memory={0: "30GB", 1: "30GB", 2: "30GB", 3: "30GB"})
        self.pipe = p.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def generate_images(self, prompt: str, n_images: int) -> Any:
        images = []
        for i in range(n_images):
            self.generator.manual_seed(i)
            image = self.pipe(prompt=prompt+tti_positive_prompt, negative_prompt=sdxl_negative_prompt, 
                              generator=self.generator, num_inference_steps=20, guidance_scale=7.0).images[0]
            images.append(image)
        return images

    def generate_single_image(self, prompt, seed: int) -> Any:
        self.generator.manual_seed(seed)
        image = self.pipe(prompt=prompt+tti_positive_prompt, negative_prompt=sdxl_negative_prompt, 
                            generator=self.generator, num_inference_steps=20, guidance_scale=7.0).images[0]
        return image

    def store_image(self, image: Any, path: str) -> None:
        image.save(path)

if __name__ == "__main__":


    from src.utils.deterministic import make_deterministic
    make_deterministic()

    tti = Sd35TextToImage()
    
    # Run 1
    r1_images = tti.generate_images("photograph of a criminal, color", n_images = 2)
    for i, image in enumerate(r1_images):
        tti.store_image(image, f"sd3_r1_{i}.png")

    r2_images = tti.generate_images("photograph of a criminal, color", n_images = 2)
    for i, image in enumerate(r2_images):
        tti.store_image(image, f"sd3_r2_{i}.png")