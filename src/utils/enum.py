from enum import Enum

class ImageType(Enum):
    GENERATED = "images"
    BIASED = "biased_images"
    CONTROL = "control_images"