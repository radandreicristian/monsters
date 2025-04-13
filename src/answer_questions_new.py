import typer
from typing import Literal
from typing_extensions import Annotated
import os
import re
import json

from src.vqa import VqaFactory
from src.vqa.blip2_cond_gen import Blip2VqaWithScoring
from src.utils.mapping import id_to_local_name
from src.logger.utils import get_logger
from src.utils.constants import FORMULATION_TYPES

logger = get_logger()

def find_yes_or_no(text):
    # Define a regex pattern to match 'yes' or 'no' irrespective of case
    pattern = re.compile(r'\b(yes|no|unknown)\b', re.IGNORECASE)
    
    # Search for the pattern in the text
    match = pattern.search(text)
    
    # If a match is found, return it in lowercase
    if match:
        return match.group().lower()
    else:
        # Raise a ValueError if neither 'yes' nor 'no' is found
        raise ValueError("No 'yes', 'no', 'unknown' not found in the text")

def answer_questions(image_generation_model_id: str,
                     vqa_model: str):
    """
    Perform VQA on images generate by a model. The images can be either control images or biased images.
    
    """
    model_name = id_to_local_name[image_generation_model_id]

    images_root_path = f"data/biased_images/{model_name}"

    logger.info(images_root_path)

    prompt_template = 'Question: Is this a {}? Answer:'
    
    with open("data/concepts.json", "r") as f:
        concepts = json.load(f)
    
    vqa = Blip2VqaWithScoring()
    

    for concept_name, concept_formulation in concepts.items():

        if concept_name != "gang_activity":
            continue

        concept = concept_formulation["direct"]
        prompt = prompt_template.format(concept)
        
        logger.info(f"Analyzing for {concept}. VQA prompt: {prompt}")
        answers = {}
        
        concept_root = os.path.join(images_root_path, concept_name, "direct")

        image_paths = [os.path.join(concept_root, file) for file in os.listdir(concept_root) if (file.endswith('.png') and not 'face' in file)]
    
        for idx, image_path in enumerate(image_paths):
            
            if idx > 5:
                continue

            answer = vqa.generate_freeform_answers(image_paths=[image_path], prompt=prompt)[0]
            logger.info(answer)
            try:
                sanitized = find_yes_or_no(answer)
                file_id = image_path.split("/")[-1].replace(".png", "").split("_")[-1]
                answers[file_id] = sanitized
            except ValueError: 
                logger.warning(f"VQA did not provide a Yes/No answer: {answer}")
        
        # groupped_answers = {}
        # for key, value in answers.items():
        #     # Convert the key to an integer
        #     int_key = int(key)
            
        #     # Check if the value is already a key in the new dictionary
        #     if value not in groupped_answers:
        #         groupped_answers[value] = []
            
        #     # Append the integer key to the list associated with the value
        #     groupped_answers[value].append(int_key)

        # answer_counts = {key: len(value) for key, value in groupped_answers.items()}

        # os.makedirs(f"{concept_root}/answers/{vqa_model}", exist_ok=True)

        # output_path = f"{concept_root}/answers/{vqa_model}/{concept_name}.json"

        # with open(output_path, "w") as f:
        #     json.dump(answer_counts, f)

        # logger.info(f"Answers for {concept_name}: {answer_counts}")

if __name__ == '__main__':
    typer.run(answer_questions)