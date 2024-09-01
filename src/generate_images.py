import json
import os

import typer

from src.tti.mock import MockTextToImage
from src.tti.stable_diffusion import StableDiffusionTextToImage
from src.logger.utils import get_logger
from diffusers.utils.testing_utils import enable_full_determinism
from diffusers.utils.logging import disable_progress_bar

import torch
import numpy as np
import random

# Deterministic kamaehama

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
enable_full_determinism()
os.env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

logger = get_logger()
disable_progress_bar()

model_mapping = {
    "dp": "digiplay/AbsoluteReality_v1.8.1"
}


def get_generator(generation_model_name: str):
    if generation_model_name == 'mock':
        return MockTextToImage()
    else:
        return StableDiffusionTextToImage(model_id=generation_model_name)


def generate_images(model_name: str,
                    prompts_root_dir: str = 'data/prompts') -> None:
    
    generation_model_name = model_mapping[model_name]
    image_generator = get_generator(generation_model_name)

    generation_model_name_path = generation_model_name.replace("/", "-")
    logger.info(image_generator)
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
                    image = image_generator.generate_image(prompt)
                    image_generator.store_image(image, image_path)


if __name__ == '__main__':
    typer.run(generate_images)
