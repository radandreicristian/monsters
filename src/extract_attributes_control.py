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
import pandas as pd

logger = get_logger()


config = {
    **dotenv_values("src/config/.env"),
    **os.environ
}

fair_face_attribute_extractor = FairFaceImageAttributeExtractor(**config)


async def process_images_for_attribute(images_root) -> None:
    fair_face_attributes, fairface_majority_attributes = await fair_face_attribute_extractor.extract_attributes(images_root)

    extracted_attributes = {
        **fairface_majority_attributes,
    }

    with open(f'{images_root}/attributes.json', 'w') as f:
        json.dump(extracted_attributes, f)
    
    attribute_breakdown = {
        'race': fair_face_attributes["race"].value_counts().to_dict(),
        'gender': fair_face_attributes["gender"].value_counts().to_dict(),
        'age': fair_face_attributes["age"].value_counts().to_dict()
    }
    
    with open(f'{images_root}/attributes_breakdown.json', 'w') as f:
        json.dump(attribute_breakdown, f)

@run_async
async def extract_attributes(model_id: str,
                             images_root: str = 'data/control_images',
                             ) -> None:
    local_model_name = id_to_local_name[model_id]
    await process_images_for_attribute(os.path.join(images_root, local_model_name))


if __name__ == '__main__':
    typer.run(extract_attributes)