import json
import os.path

import typer

from src.logger import get_logger

logger = get_logger()


def generate_prompts(attributes_path: str = "data/attributes.json",
                     templates_path: str = "data/templates.json") -> None:
    with open(attributes_path, "r") as f:
        attributes = json.load(f)

    with open(templates_path, "r") as f:
        templates = json.load(f)
    total_prompts = 0
    # e.g. "crime": {...}
    for group_name, group_values in attributes.items():

        # e.g. "felony": {...}
        for subgroup_name, formulations in group_values.items():
            dir_path = os.path.join("data", "prompts", group_name)
            os.makedirs(dir_path, exist_ok=True)

            attribute_value_filename = f"{subgroup_name}.json"
            file_path = os.path.join(dir_path, attribute_value_filename)
            prompts = []

            # e.g. {"label": "criminal", "description": "commited a crime"}
            for formulation_type, val in formulations.items():

                formulation_templates = templates[formulation_type]
                prompts = prompts + [template.replace("[FILL]", val) for template in formulation_templates]
                total_prompts += len(prompts)
            with open(file_path, "w") as f:
                json.dump(prompts, f, indent=4)
    logger.info(f"Total prompts {total_prompts}")


if __name__ == '__main__':
    typer.run(generate_prompts)
