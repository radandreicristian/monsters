import torch
from diffusers import StableDiffusionPipeline, KDPM2DiscreteScheduler
from src.tti.base import BaseTextToImage

from typing import Any
from src.logger.utils import get_logger

logger = get_logger()

class AbsoluteRealityTextToImage(BaseTextToImage):

    def __init__(self, 
                 *args, **kwargs):
        
        self.generator = torch.Generator(device="cuda")
        self.pipe = StableDiffusionPipeline.from_pretrained('digiplay/AbsoluteReality_v1.8.1', torch_dtype=torch.float16).to("cuda")
        self.pipe.scheduler = KDPM2DiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

    def generate_images(self, prompt: str, n_images: int) -> Any:
        images = []
        for i in range(n_images):
            self.generator.manual_seed(i)
            image = self.pipe(prompt=prompt, generator=self.generator).images[0]
            images.append(image)
        return images

    def store_image(self, image: Any, path: str) -> None:
        image.save(path)

if __name__ == "__main__":
    from src.utils.deterministic import make_deterministic
    make_deterministic()

    tti = AbsoluteRealityTextToImage()
    
    # Run 1
    r1_images = tti.generate_images("Portrait of a criminal", n_images = 2)
    for i, image in enumerate(r1_images):
        tti.store_image(image, f"sdxl_r1_{i}.png")

    r2_images = tti.generate_images("Portrait of a criminal", n_images = 2)
    for i, image in enumerate(r2_images):
        tti.store_image(image, f"sdxl_r2_{i}.png")