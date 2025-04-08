import typer
from typing import Literal
from typing_extensions import Annotated
import os
import re
import json

from src.vqa.blip_cond_gen import BlipVqaWithScoring
from src.vqa.blip2_cond_gen import Blip2VqaWithScoring
from src.utils.mapping import id_to_local_name
from src.logger.utils import get_logger
from src.utils.constants import FORMULATION_TYPES

logger = get_logger()


def answer_questions(image_generation_model_id: str,
                     vqa_model: str):

    model_name = id_to_local_name[image_generation_model_id]
    images_root_path = f"data/control_images/{model_name}"

    logger.info(images_root_path)

    prompt_template = 'Is the person in the image a {}? Answer with YES or NO.'

    with open("data/concepts.json", "r") as f:
        concepts = json.load(f)

    vqa = Blip2VqaWithScoring()
    BATCH_SIZE = 4

    image_paths = sorted([
        os.path.join(images_root_path, file)
        for file in os.listdir(images_root_path)
        if file.endswith('.png') and 'face' not in file
    ])

    for concept_name, concept_formulation in concepts.items():

        if concept_name != "crime":
            continue

        concept = concept_formulation["direct"]
        prompt = prompt_template.format(concept)
        logger.info(f"Analyzing for {concept_name}. VQA prompt: {prompt}")

        answers = []

        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            batch_scores = vqa.score_answers(batch_paths, prompt)

            for path, score in zip(batch_paths, batch_scores):
                score["path"] = path
                answers.append(score)

        total_yes = sum(1 for x in answers if x["yes"] > 0.5)
        total_no = len(answers) - total_yes

        output_data = {
            "total_yes": total_yes,
            "total_no": total_no,
            "each": answers
        }

        output_dir = os.path.join(images_root_path, "answers", vqa_model)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{concept}.json")

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Answers for {concept}: YES={total_yes}, NO={total_no}")
        
if __name__ == '__main__':
    typer.run(answer_questions)