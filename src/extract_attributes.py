import json
import os
import asyncio

import typer

from src.utils.async_wrapper import run_async
from src.image.openai import OpenAiImageAttributeExtractor
from src.image.fairface import FairFaceImageAttributeExtractor
from src.utils.short_circuit import short_circuit_concept
from dotenv import dotenv_values

from src.logger.utils import get_logger
from src.utils.mapping import id_to_local_name

logger = get_logger()


config = {
    **dotenv_values("src/config/.env"),
    **dotenv_values("src/config/.secret.env"),
    **os.environ
}

openai_image_attribute_extractor = OpenAiImageAttributeExtractor(**config)
fair_face_attribute_extractor = FairFaceImageAttributeExtractor(**config)


async def process_images_for_attribute(images_root) -> None:
    openai_attributes = await openai_image_attribute_extractor.extract_attributes(images_root)
    fair_face_attributes = await fair_face_attribute_extractor.extract_attributes(images_root)

    extracted_attributes = {
        **fair_face_attributes,
        "attributes": openai_attributes
    }

    deconstructed_path = os.path.normpath(images_root).split(os.sep)

    # deconstructed_path like ['data', 'images', 'sd3', 'drug_trafficknig', 'direct']
    # attributes_path like ['data', 'attributes', 'sd3', 'drug_trafficking', 'direct']
    attributes_path = os.sep.join([*deconstructed_path[:1],
                                  "attributes",
                                  *deconstructed_path[2:]])
    logger.info(attributes_path)
    os.makedirs(attributes_path, exist_ok=True)
    with open(f'{attributes_path}/attributes.json', 'w') as f:
        json.dump(extracted_attributes, f)


@run_async
async def extract_attributes(model_id: str,
                             images_root: str = 'data/images',
                             short_circuit: bool = False) -> None:
    local_model_name = id_to_local_name[model_id]
    if short_circuit:
        tasks = [process_images_for_attribute(os.path.join(images_root, local_model_name, short_circuit_concept, formulation)) for formulation in ("direct", "indirect")]
    else:
        tasks = [process_images_for_attribute(os.path.join(images_root, local_model_name, attribute, formulation))
                for attribute in os.listdir(os.path.join(images_root, local_model_name))
                for formulation in ("direct", "indirect")]
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    typer.run(extract_attributes)
