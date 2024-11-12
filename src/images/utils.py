from pathlib import Path
from typing import Union

import numpy as np
import PIL


def resize_image(img: np.ndarray, height: int = 360, width: int = 360) -> np.ndarray:
    img = PIL.Image.fromarray(img)
    img = img.resize((height, width))
    return np.array(img)


def load_and_process_image(
    img_path: Union[str, Path],
    height: int = 360,
    width: int = 360,
) -> np.ndarray:
    img = PIL.Image.open(img_path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((height, width))

    return np.array(img)
