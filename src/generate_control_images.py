import json
import os

import typer
from src.tti import *
from src.logger.utils import get_logger
from src.utils.deterministic import make_deterministic

from src.logger.utils import get_logger
from src.utils.deterministic import make_deterministic
from src.utils.short_circuit import short_circuit_concept
from diffusers.utils.logging import disable_progress_bar

from src.utils.mapping import id_to_local_name
make_deterministic()
disable_progress_bar()

logger = get_logger()

def generate_images(model_id: str,
                    images_per_prompt: int = 100,
                   ) -> None:
    
    text_to_image = name_to_tti_mapping[model_id]()

    generation_model_local_name = id_to_local_name[model_id]

    prompts = ["photo of a person"]
   
    image_dir = os.path.join('data', 'control_images', generation_model_local_name)
    os.makedirs(image_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        logger.info(prompt)
        metadata_path = os.path.join(image_dir, f"p_{i}.json")
        with open(metadata_path, "w") as f:
            json.dump({"prompt": prompt}, f, indent=4)
        # images = text_to_image.generate_images(prompt, n_images=images_per_prompt)

        for seed in range(images_per_prompt):
            image_path = os.path.join(image_dir, f"p_{i}_seed_{seed}.png")
            # Re-generate the image only if force_override is not set
            if os.path.exists(image_path):
                logger.info("Skipping existing image")
                continue
            image = text_to_image.generate_single_image(prompt, seed)
            text_to_image.store_image(image, image_path)

if __name__ == '__main__':
    typer.run(generate_images)
