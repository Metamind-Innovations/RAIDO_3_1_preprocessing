from PIL import Image
import numpy as np


def load_image(img_path: str) -> np.ndarray:
    image = Image.open(img_path)
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    return image


def image_path2id(img_json: dict) -> dict:
    path_to_id = {}
    for idx, path in enumerate(img_json["image_paths"]):
        path_to_id[path] = idx
    img_json["path_to_id"] = path_to_id

    return img_json


def resize_image(img: np.ndarray, height: int = 360, width: int = 360) -> np.ndarray:
    img = Image.fromarray(img)
    img = img.resize((height, width))
    return np.array(img)


# TODO: Maybe this has to change
# def load_and_process_image(
#     img_path: Union[str, Path],
#     height: int = 360,
#     width: int = 360,
# ) -> np.ndarray:
#     img = PIL.Image.open(img_path)

#     if img.mode != "RGB":
#         img = img.convert("RGB")

#     img = img.resize((height, width))

#     return np.array(img)



