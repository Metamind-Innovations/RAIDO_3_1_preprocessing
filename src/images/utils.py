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


def normalize_pixel_values(
    image: np.ndarray,
) -> np.ndarray:
    if image.ndim == 3 and image.shape[-1] == 4:  # RGBA
        rgb = image[..., :3]
        alpha = image[..., 3]

        if rgb.max() > 1.0:
            normalized_rgb = rgb / 255.0
        else:
            normalized_rgb = rgb.copy()

        normalized_image = np.dstack((normalized_rgb, alpha))
    else:  # RGB or grayscale
        if image.max() > 1.0:
            normalized_image = image / 255.0
        else:
            normalized_image = image.copy()

    return normalized_image


def calculate_normalization_stats(image_paths: list) -> dict:
    """Calculate mean and std across a dataset of images."""
    means = []
    stds = []

    for path in image_paths:
        image = load_image(path)
        # Convert to float and scale to [0,1]
        image = image.astype(np.float32) / 255.0

        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:  # RGBA
            image = image[..., :3]  # Keep only RGB channels

        # Calculate per-channel statistics
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))

        means.append(mean)
        stds.append(std)

    # Convert lists to numpy arrays before calculating final statistics
    means = np.array(means)
    stds = np.array(stds)

    return {
        "mean": means.mean(axis=0).tolist(),
        "std": stds.mean(axis=0).tolist(),
    }
