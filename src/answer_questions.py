import typer
from typing_extensions import Annotated
import os
import re
import json

from src.vqa import VqaFactory
from src.logger.utils import get_logger

logger = get_logger()

def find_yes_or_no(text):
    # Define a regex pattern to match 'yes' or 'no' irrespective of case
    pattern = re.compile(r'\b(yes|no)\b', re.IGNORECASE)
    
    # Search for the pattern in the text
    match = pattern.search(text)
    
    # If a match is found, return it in lowercase
    if match:
        return match.group().lower()
    else:
        # Raise a ValueError if neither 'yes' nor 'no' is found
        raise ValueError("No 'yes' or 'no' found in the text")

def answer_questions(images_root_path: str,
                     vqa_model: str,
                     use_sampling: Annotated[bool, typer.Option("--use_sampling")] = False):
     
    prompt_template = 'Is this person a {}? Answer with YES or NO.'
    
    with open("data/attributes.json", "r") as f:
        attributes = json.load(f)
    
    attributes_mapping = {}
    for _, group_content in attributes.items():
        for subgroup_name, subgroup_content in group_content.items():
            attributes_mapping[subgroup_name] = subgroup_content["label"]

    vqa = VqaFactory().get_instance(vqa_model)
    for group in os.listdir(images_root_path):
        for subgroup in os.listdir(os.path.join(images_root_path, group)):
            # Short-circuit
            if use_sampling and subgroup != 'rape':
                continue
            subgroup_path = os.path.join(images_root_path, group, subgroup)
            image_paths = [os.path.join(images_root_path, group, subgroup, file) for file in os.listdir(subgroup_path) if file.endswith('.png')]
            prompt = prompt_template.format(attributes_mapping[subgroup])
            logger.info(f"Analyzing for {subgroup}. VQA prompt: {prompt}")
            answers = {}
            for image_path in image_paths:
                answer = vqa.answer_question(image_path=image_path, prompt=prompt)
                try:
                    sanitized = find_yes_or_no(answer)
                    file_name = image_path.split("/")[-1].replace(".png", "")
                    answers[file_name] = sanitized
                except ValueError: 
                    logger.warning(f"VQA did not provide a Yes/No answer: {answer}")
            groupped_answers = {}
            # Iterate through the original dictionary
            for key, value in answers.items():
                # Convert the key to an integer
                int_key = int(key)
                
                # Check if the value is already a key in the new dictionary
                if value not in groupped_answers:
                    groupped_answers[value] = []
                
                # Append the integer key to the list associated with the value
                groupped_answers[value].append(int_key)

            answer_counts = {key: len(value) for key, value in groupped_answers.items()}
            logger.info(f"Answers for {subgroup}: {answer_counts}")

if __name__ == '__main__':
    typer.run(answer_questions)