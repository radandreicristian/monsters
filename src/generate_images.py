import json
import os

import typer

from src.tti.mock import MockTextToImage
from src.tti.absolute_reality import AbsoluteRealityTextToImage
from src.tti.sdxl import SdxlTextToImage
from src.logger.utils import get_logger
from src.utils.deterministic import make_deterministic
from diffusers.utils.logging import disable_progress_bar

make_deterministic()
disable_progress_bar()

logger = get_logger()

id_to_checkpoint = {
    "dp": "digiplay/AbsoluteReality_v1.8.1",
    "sdxl": "stabilityai/sdxl-turbo"
}

name_to_tti_mapping = {
    "dp": AbsoluteRealityTextToImage,
    "sdxl": SdxlTextToImage,
    "mock": MockTextToImage
}

def generate_images(model_id: str,
                    images_per_prompt: int = 1,
                    prompts_root_dir: str = 'data/prompts') -> None:
    
    checkpoint_name = id_to_checkpoint[model_id]
    text_to_image = name_to_tti_mapping[model_id]()

    generation_model_name_path = checkpoint_name.replace("/", "-")
    groups = os.listdir(prompts_root_dir)
    for group in groups:
        group_path = os.path.join(prompts_root_dir, group)
        subgroups = os.listdir(group_path)
        for subgroup in subgroups:
            logger.info(f'Generating images for {group}/{subgroup}...')
            subgroup_name = subgroup.replace(".json", "")
            subgroup_path = os.path.join(group_path, subgroup)
            with open(subgroup_path, "r") as f:
                all_prompts = json.load(f)
            for prompt_type, prompts in all_prompts.items():
                for i, prompt in enumerate(prompts):
                    image_dir = os.path.join("data", "images", generation_model_name_path, group, subgroup_name, prompt_type)
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f"{i}.png")
                    metadata_path = os.path.join(image_dir, f"{i}.json")
                    with open(metadata_path, "w") as f:
                        json.dump({"prompt": prompt}, f, indent=4)
                    images = text_to_image.generate_images(prompt, n_images=images_per_prompt)
                    for image in images:
                        text_to_image.store_image(image, image_path)


if __name__ == '__main__':
    typer.run(generate_images)
