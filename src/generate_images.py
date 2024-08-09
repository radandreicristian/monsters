import json
import os

import typer

from src.tti.mock import MockTextToImage

image_generator = MockTextToImage()


def generate_images(generation_model_name: str,
                    prompts_root_dir: str = 'data/prompts') -> None:
    groups = os.listdir(prompts_root_dir)
    for group in groups:
        group_path = os.path.join(prompts_root_dir, group)
        subgroups = os.listdir(group_path)
        for subgroup in subgroups:
            subgroup_name = subgroup.replace(".json", "")
            subgroup_path = os.path.join(group_path, subgroup)
            with open(subgroup_path, "r") as f:
                prompts = json.load(f)
            for i, prompt in enumerate(prompts):
                image_dir = os.path.join("data", "images", generation_model_name, group, subgroup_name)
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join("data", "images", generation_model_name, group, subgroup_name, f"{i}.png")
                image = image_generator.generate_image(prompt)
                image_generator.store_image(image, image_path)


if __name__ == '__main__':
    typer.run(generate_images)
