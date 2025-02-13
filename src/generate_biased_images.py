import json
import os

import typer
from src.tti import name_to_tti_mapping
from src.logger.utils import get_logger
from src.utils.deterministic import make_deterministic
from src.utils.short_circuit import short_circuit_concept
from diffusers.utils.logging import disable_progress_bar

from src.utils.mapping import id_to_hf_checkpoint, id_to_local_name
make_deterministic()
disable_progress_bar()

logger = get_logger()


def generate_images(model_id: str,
                    images_per_prompt: int = 5,
                    short_circuit: bool = True) -> None:
    
    text_to_image = name_to_tti_mapping[model_id]()

    generation_model_local_name = id_to_local_name[model_id]

    biased_prompts_root = f"data/biased_prompts/{generation_model_local_name}"

    concepts = os.listdir(biased_prompts_root)

    for concept in concepts:

        if short_circuit and concept != short_circuit_concept:
            continue
        
        for formulation in ("direct", "indirect"):

            with open(os.path.join(biased_prompts_root, concept, formulation, "biased_generation_prompts.json"), 'r') as f:

                biased_generation_promps = json.load(f) 
            
            for i, prompt in enumerate(biased_generation_promps):
                logger.info(f"Generating images for prompt {prompt}")
                
                image_dir = os.path.join("data", "biased_images", generation_model_local_name, concept, formulation)
                
                os.makedirs(image_dir, exist_ok=True)
                
                metadata_path = os.path.join(image_dir, f"p_{i}.json")
                
                with open(metadata_path, "w") as f:
                    json.dump({"prompt": prompt}, f, indent=2)
                
                images = text_to_image.generate_images(prompt, n_images=images_per_prompt)
                
                for seed_index, image in enumerate(images):
                    image_path = os.path.join(image_dir, f"p_{i}_seed_{seed_index}.png")
                    text_to_image.store_image(image, image_path)


if __name__ == '__main__':
    typer.run(generate_images)
