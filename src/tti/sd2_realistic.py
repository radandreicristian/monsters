import torch
from diffusers import StableDiffusionPipeline, KDPM2DiscreteScheduler
from src.tti.base import BaseTextToImage

from typing import Any
from src.logger.utils import get_logger
from src.utils.prompts import tti_negative_prompt, tti_positive_prompt

logger = get_logger()

class SdTwoRealisticTextToImage(BaseTextToImage):

    def __init__(self, 
                 *args, **kwargs):
        
        self.generator = torch.Generator(device="cuda")
        self.pipe = StableDiffusionPipeline.from_pretrained('friedrichor/stable-diffusion-2-1-realistic', torch_dtype=torch.float16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def generate_images(self, prompt: str, n_images: int) -> Any:
        images = []
        for i in range(n_images):
            self.generator.manual_seed(i)
            # prompt = prompt + tti_positive_prompt
            image = self.pipe(prompt=prompt, negative_prompt=tti_negative_prompt, num_inference_steps=20, generator=self.generator).images[0]
            # image = self.pipe(prompt=prompt, num_inference_steps=20, generator=self.generator).images[0]
            images.append(image)
        return images

    def store_image(self, image: Any, path: str) -> None:
        image.save(path)

if __name__ == "__main__":
    from src.utils.deterministic import make_deterministic
    make_deterministic()

    tti = SdTwoRealisticTextToImage()
    
    # Run 1
    r1_images = tti.generate_images("Portrait of a criminal", n_images = 2)
    for i, image in enumerate(r1_images):
        tti.store_image(image, f"sd2r_r1_{i}.png")

    r2_images = tti.generate_images("Portrait of a criminal", n_images = 2)
    for i, image in enumerate(r2_images):
        tti.store_image(image, f"sd2r_r2_{i}.png")