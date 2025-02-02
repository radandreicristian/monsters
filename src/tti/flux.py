import torch
from diffusers import FluxPipeline
from src.tti.base import BaseTextToImage
from src.utils.prompts import tti_negative_prompt, tti_positive_prompt
from typing import Any
from src.logger.utils import get_logger

logger = get_logger()



class FluxTextToImage(BaseTextToImage):

    def __init__(self, 
                 *args, **kwargs):

        # Black magic from https://github.com/Lightning-AI/litgpt/issues/327#issuecomment-1664674460
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

        self.generator = torch.Generator(device="cuda")
        
        self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", 
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="balanced",
                                                 max_memory={0: "32GB", 1: "32GB", 2: "32GB"})

    def generate_images(self, prompt: str, n_images: int) -> Any:
        images = []
        for i in range(n_images):
            self.generator.manual_seed(i)
            image = self.pipe(prompt=prompt+tti_positive_prompt,
                              generator=self.generator, 
                              height=512, width=512, 
                              guidance_scale=0.0, 
                              num_inference_steps=2).images[0]
            images.append(image)
        return images

    def store_image(self, image: Any, path: str) -> None:
        image.save(path)

if __name__ == "__main__":

    from src.utils.deterministic import make_deterministic

    make_deterministic()
    
    tti = FluxTextToImage(model_id='black-forest-labs/FLUX.1-schnell')
    
    # Run 1
    r1_images = tti.generate_images("Portrait of a criminal", n_images = 2)
    for i, image in enumerate(r1_images):
        tti.store_image(image, f"flux_r1_{i}.png")

    r2_images = tti.generate_images("Portrait of a criminal", n_images = 2)
    for i, image in enumerate(r2_images):
        tti.store_image(image, f"flux_r2_{i}.png")