import json
import os
import asyncio

import numpy as np
import typer

from src.utils.async_wrapper import run_async
from src.image.openai import OpenAiImageAttributeExtractor
from src.image.fairface import FairFaceImageAttributeExtractor
from src.utils.short_circuit import short_circuit_concept
from dotenv import dotenv_values

from src.logger.utils import get_logger
from src.utils.mapping import id_to_local_name
import pandas as pd

from PIL import Image
from src.utils.constants import FORMULATION_TYPES

logger = get_logger()


config = {
    **dotenv_values("src/config/.env"),
    **dotenv_values("src/config/.secret.env"),
    **os.environ
}

@run_async
async def extract_attributes(model_id: str,
                             images_root: str = 'data/concept_images',
                             short_circuit: bool = False,
                             ) -> None:
    images_for_model_root = f"{images_root}/{id_to_local_name[model_id]}"
    
    for concept in os.listdir(images_for_model_root):
        logger.info(f"Generating mean face for {concept}")

        for formulation_type in FORMULATION_TYPES:
            images_root = os.path.join(images_for_model_root, concept, formulation_type)

            faces = [os.path.join(images_root, image) for image in os.listdir(images_root) if "_face_" in image]

            sum = None
            count = 0

            for face in faces:
                img_array = np.array(Image.open(face).convert("RGB"), dtype=np.float64)
                if sum is None:
                    sum = img_array
                else:
                    sum += img_array
                count+=1
            
            mean_array = (sum/count).astype(np.uint8) # type: ignore
            mean_image = Image.fromarray(mean_array, mode="RGB")

            output_path = os.path.join(images_root, "mean_face_new.png")
            mean_image.save(output_path)

if __name__ == '__main__':
    typer.run(extract_attributes)
