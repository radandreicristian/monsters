import torch
from diffusers import StableDiffusionPipeline, KDPM2DiscreteScheduler
from diffusers.utils.testing_utils import enable_full_determinism
from diffusers.utils.logging import disable_progress_bar
from src.tti.base import BaseTextToImage

from typing import Any
from src.logger.utils import get_logger

logger = get_logger()

class StableDiffusionTextToImage(BaseTextToImage):

    def __init__(self, 
                 model_id: str,
                 *args, **kwargs):
        
        disable_progress_bar()
        enable_full_determinism()

        generator = torch.Generator(device="cpu").manual_seed(21)

        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, generator=generator)
        self.pipe.scheduler = KDPM2DiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def generate_image(self, prompt: str) -> Any:
        image = self.pipe(prompt).images[0]
        return image

    def store_image(self, image: Any, path: str) -> None:
        image.save(path)
