import torch
from diffusers import FluxPipeline
from src.tti.base import BaseTextToImage
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

from typing import Any
from src.logger.utils import get_logger

logger = get_logger()



class FluxTextToImage(BaseTextToImage):

    def __init__(self, 
                 model_id: str,
                 *args, **kwargs):
        
        self.generator = torch.Generator(device="cuda")
        with init_empty_weights():
            self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = load_checkpoint_and_dispatch(
            self.pipe, 
            checkpoint=model_id,
            device_map='auto', 
            dtype=torch.float16
        )

    def generate_images(self, prompt: str, n_images: int) -> Any:
        images = []
        for i in range(n_images):
            self.generator.manual_seed(i)
            image = self.pipe(prompt=prompt, generator=self.generator, height=512, width=512, guidance_scale=0.0, num_inference_steps=1).images[0]
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
        tti.store_image(image, f"flux/r1_{i}.png")

    r2_images = tti.generate_images("Portrait of a criminal", n_images = 2)
    for i, image in enumerate(r2_images):
        tti.store_image(image, f"flux/r2_{i}.png")