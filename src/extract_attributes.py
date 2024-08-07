import os

from src.image.openai import OpenAiImageAttributeExtractor
from dotenv import dotenv_values


config = {
    **dotenv_values("src/config/.env"),
    **dotenv_values("src/config/.secret.env"),
    **os.environ
}

if __name__ == '__main__':
    openai_image_attribute_extractor = OpenAiImageAttributeExtractor(**config)
    response = openai_image_attribute_extractor.extract_attributes(images_root="data/images/sex_abuse/child_abuse")
    print(response)
