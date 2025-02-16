import json
import os

import typer
from src.tti import *
from src.logger.utils import get_logger
from src.utils.deterministic import make_deterministic
from diffusers.utils.logging import disable_progress_bar


logger = get_logger()

id_to_checkpoint = {
    "dp": "digiplay/AbsoluteReality_v1.8.1",
    "sdxl": "stabilityai/sdxl-turbo",
    "sd2r": "friedrichor/stable-diffusion-2-1-realistic",
    "flux": "black-forest-labs/FLUX.1-schnell"
}

name_to_tti_mapping = {
    "dp": AbsoluteRealityTextToImage,
    "sdxl": SdxlTextToImage,
    "mock": MockTextToImage,
    "sd2r": SdTwoRealisticTextToImage,
    "flux": FluxTextToImage
}

make_deterministic()

def generate_images(model_id: str,
                    images_per_prompt: int = 25,
                   ) -> None:
    
    checkpoint_name = id_to_checkpoint[model_id]
    text_to_image = name_to_tti_mapping[model_id]()

    generation_model_name_path = checkpoint_name.replace("/", "-")

    prompts = {
        "photo of a person",
        "picture of a person",
        "face shot of a person",
        "photograph of a peson"
    }

    logger.info(prompts)
    text_to_image = name_to_tti_mapping[model_id]()
   
    image_dir = os.path.join('data', 'control_images', generation_model_name_path)
    os.makedirs(image_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        metadata_path = os.path.join(image_dir, f"p_{i}.json")
        with open(metadata_path, "w") as f:
            json.dump({"prompt": prompt}, f, indent=4)
        images = text_to_image.generate_images(prompt, n_images=images_per_prompt)
        for seed_index, image in enumerate(images):
            image_path = os.path.join(image_dir, f"p_{i}_seed_{seed_index}.png")
            text_to_image.store_image(image, image_path)

if __name__ == '__main__':
    typer.run(generate_images)
