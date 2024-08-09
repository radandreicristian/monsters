import json
import os
import asyncio

from src.image.openai import OpenAiImageAttributeExtractor
from src.image.fairface import FairFaceImageAttributeExtractor
from dotenv import dotenv_values


config = {
    **dotenv_values("src/config/.env"),
    **dotenv_values("src/config/.secret.env"),
    **os.environ
}

openai_image_attribute_extractor = OpenAiImageAttributeExtractor(**config)
fair_face_attribute_extractor = FairFaceImageAttributeExtractor(**config)


async def process_group_images(subgroup_root) -> None:
    openai_attributes = await openai_image_attribute_extractor.extract_attributes(subgroup_root)
    fair_face_attributes = await fair_face_attribute_extractor.extract_attributes(subgroup_root)

    extracted_attributes = {
        **fair_face_attributes,
        "attributes": openai_attributes
    }

    deconstructed_path = os.path.normpath(subgroup_root).split(os.sep)
    attributes_path = os.sep.join([*deconstructed_path[:-3],
                                    "attributes",
                                   *deconstructed_path[-2:]])
    os.makedirs(attributes_path, exist_ok=True)
    with open(f'{attributes_path}/attributes.json', 'w') as f:
        json.dump(extracted_attributes, f)


async def extract_attributes(images_root: str):
    tasks = [process_group_images(os.path.join(images_root, group, subgroup)) for group in os.listdir(images_root)
             for subgroup in os.listdir(os.path.join(images_root, group))
             if subgroup == "child_abuse"]
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    root = 'data/images'
    asyncio.run(extract_attributes(root))