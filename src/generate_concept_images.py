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
                    prompts_root_dir: str = 'data/prompts',
                    short_circuit: bool = True) -> None:
    
    text_to_image = name_to_tti_mapping[model_id]()

    generation_model_name_path = id_to_local_name[model_id]
    concepts = sorted(os.listdir(prompts_root_dir))
    for concept in concepts:

        # If short_circuit is enabled, generate images for a single subgroup only (test mode)
        concept_name = concept.replace(".json", "")
        if short_circuit and not (concept_name == short_circuit_concept):
            continue

        concept_prompts_path = os.path.join(prompts_root_dir, concept)
        logger.info(f'Generating images for {concept_name} in {concept_prompts_path}')

        with open(concept_prompts_path, "r") as f:
            all_prompts = json.load(f)
        
        for formulation_type, prompts in all_prompts.items():
            for i, prompt in enumerate(prompts):
                logger.info(f"Generating images for prompt {prompt}")
                image_dir = os.path.join("data", "concept_images", generation_model_name_path, concept_name, formulation_type)
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
