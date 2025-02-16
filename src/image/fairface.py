import os
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import dlib
import asyncio
from src.image.base import BaseImageAttributeExtractor
from torchvision.models import ResNet34_Weights
from src.logger.utils import get_logger

logger = get_logger()


class FairFaceImageAttributeExtractor(BaseImageAttributeExtractor):

    def __init__(self, *args, **kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_fair_face_model('src/models/res34_fair_align_multi_7_20190809.pt', 18)
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.labels = {
            "race": ['caucasian', 'afro american', 'latino-hispanic',
                     'east asian', 'southeast asian', 'indian', 'middle eastern'],
            "gender": ['male', 'female'],
            "age": ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
            # "age": ["infant", "child", "teenager", "young adult", "adult", "middle-aged", "mature", "senior", "elderly"] # noqa
        }
        self.face_detector = dlib.cnn_face_detection_model_v1('src/models/mmod_human_face_detector.dat') # type: ignore
        self.shape_predictor = dlib.shape_predictor('src/models/shape_predictor_5_face_landmarks.dat') # type: ignore

    def load_fair_face_model(self, model_path, num_classes):
        model = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model = model.to(self.device)
        model.eval()
        return model

    async def detect_faces(self,
                           image_paths: list[str],
                           default_max_size: int = 800,
                           size: int = 300,
                           padding: float = 0.25):
        tasks = [self.process_image(image_path, default_max_size, size, padding) for image_path in image_paths]
        await asyncio.gather(*tasks)

    async def process_image(self, image_path: str, default_max_size, size, padding):
        img = dlib.load_rgb_image(image_path) # type: ignore
        old_height, old_width, _ = img.shape
        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height = int(default_max_size * old_width / old_height), default_max_size
        img = dlib.resize_image(img, rows=new_height, cols=new_width) # type: ignore
        detected_faces = self.face_detector(img, 1)
        num_faces = len(detected_faces)

        if num_faces == 0:
            logger.warning(f"No faces found in '{image_path}'")
            return

        faces = dlib.full_object_detections() # type: ignore

        for detected_face in detected_faces:
            rect = detected_face.rect
            faces.append(self.shape_predictor(img, rect))
        images = dlib.get_face_chips(img, faces, size=size, padding=padding) # type: ignore

        directory, file_name = os.path.split(image_path)

        for idx, image in enumerate(images):
            path_sp = os.path.splitext(file_name)
            detected_face_path = os.path.join(directory, f"{path_sp[0]}_face_{idx}{path_sp[1]}")
            dlib.save_image(image, detected_face_path) # type: ignore

    async def predict_attributes(self, image_paths: list[str]):
        tasks = [self.process_prediction(img_name) for img_name in image_paths]
        results = await asyncio.gather(*tasks)

        result_df = pd.DataFrame(results, columns=['face_name_align',
                                                   'race_preds', 'gender_preds', 'age_preds',
                                                   'race_scores', 'gender_scores', 'age_scores'])

        result_df['race'] = result_df['race_preds'].apply(lambda x: self.labels["race"][x])
        result_df['gender'] = result_df['gender_preds'].apply(lambda x: self.labels["gender"][x])
        result_df['age'] = result_df['age_preds'].apply(lambda x: self.labels["age"][x])
        return result_df[['race', 'gender', 'age']]

    async def process_prediction(self, image_path):
        image = dlib.load_rgb_image(image_path) # type: ignore
        image = self.trans(image)
        image = image.view(1, 3, 224, 224)
        image = image.to(self.device)

        outputs = self.model(image).cpu().detach().numpy().squeeze()

        race_scores = np.exp(outputs[:7]) / np.sum(np.exp(outputs[:7]))
        gender_scores = np.exp(outputs[7:9]) / np.sum(np.exp(outputs[7:9]))
        age_scores = np.exp(outputs[9:18]) / np.sum(np.exp(outputs[9:18]))

        race_preds = np.argmax(race_scores)
        gender_preds = np.argmax(gender_scores)
        age_preds = np.argmax(age_scores)

        return [image_path,
                race_preds, gender_preds, age_preds,
                race_scores, gender_scores, age_scores]

    async def extract_attributes(self, images_root: str) -> tuple[pd.DataFrame, dict[str, str]]:
        valid_image_paths = [os.path.join(images_root, x)
                             for x in os.listdir(images_root)
                             if x.endswith(('.jpg', '.png'))
                             and "face" not in x]  # Exclude processed images
        await self.detect_faces(valid_image_paths)
        processed_image_paths = [os.path.join(images_root, x) for x in os.listdir(images_root) if "face" in x]
        attributes = await self.predict_attributes(image_paths=processed_image_paths)
        majority_attributes = self.get_majority_attributes(attributes)
        logger.info(majority_attributes)
        return attributes, majority_attributes

    @staticmethod
    def get_majority_attributes(results) -> dict:
        # Count the occurrences of each value
        races = results['race'].to_list()
        genders = results['gender'].to_list()
        ages = results['age'].to_list()
        race_counter = Counter(races)
        gender_counter = Counter(genders)
        age_counter = Counter(ages)

        # Find the majority value if it exists
        def find_majority(counter):
            total_items = sum(counter.values())
            for key, count in counter.items():
                if count > total_items / 2:
                    return key
            return None

        majority_race = find_majority(race_counter)
        majority_gender = find_majority(gender_counter)
        majority_age = find_majority(age_counter)
        return {'race': majority_race, 'gender': majority_gender, 'age': majority_age}
