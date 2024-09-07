import json
import os

import typer

from src.tti.mock import MockTextToImage
from src.tti.stable_diffusion import StableDiffusionTextToImage
from src.logger.utils import get_logger


logger = get_logger()


def get_generator(generation_model_name: str):
    if generation_model_name == 'mock':
        return MockTextToImage()
    else:
        return StableDiffusionTextToImage(model_id=generation_model_name)


def generate_images(generation_model_name: str) -> None:
    
    with open('data/templates.json', 'r') as f:
        templates = json.load(f)

    description_templates_mapped = [t.replace(" that [FILL]", "") for t in templates["description"]]
    label_templates_mapped = [t.replace("[FILL]", "person") for t in templates["label"]]
    
    control_templates = description_templates_mapped + label_templates_mapped
    generation_model_name_path = generation_model_name.replace("/", "-")

    logger.info(control_templates)
    image_generator = get_generator(generation_model_name)
    image_dir = os.path.join('data', 'control_images', generation_model_name_path)
    os.makedirs(image_dir, exist_ok=True)
    current = 0
    for template in control_templates:
        for _ in range(10):
            logger.info(f"Generating image for: {template}")
            image_path = os.path.join(image_dir, f'{current}.png')
            image = image_generator.generate_image(template)
            image_generator.store_image(image, image_path)
            current += 1

if __name__ == '__main__':
    typer.run(generate_images)
