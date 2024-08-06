import json
import os

import numpy as np
from PIL import Image

import typer


def generate_images(prompts_root_dir: str = 'data/prompts') -> None:
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
                array = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
                image = Image.fromarray(array)
                image_dir = os.path.join("data", "images", group, subgroup_name)
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join("data", "images", group, subgroup_name, f"{i}.png")
                image.save(image_path)


if __name__ == '__main__':
    typer.run(generate_images)
