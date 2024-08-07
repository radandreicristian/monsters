import base64


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoding = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoding}"
