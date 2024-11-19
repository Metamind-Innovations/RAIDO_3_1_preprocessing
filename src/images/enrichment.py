from typing import Union

import albumentations as A
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


# TODO: Maybe in next version add the rest of the parameters for Rotate
def rotate_image(
    images: Union[list[np.ndarray], list[Image.Image]],
    negative_angle: int = -90,
    positive_angle: int = 90,
) -> list[np.ndarray]:
    transform = A.Rotate(limit=(negative_angle, positive_angle), p=1.0)

    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        transformed_image = transform(image=image)["image"]
        transformed_images.append(transformed_image)

    return transformed_images


# TODO: can we make it more general? Can we use this more general function in the API endpoints?
def visualize_rotated_images(
    images: Union[list[np.ndarray], list[Image.Image]],
    negative_angle: int = -90,
    positive_angle: int = 90,
) -> None:
    rotated_images = rotate_image(images, negative_angle, positive_angle)

    num_images = len(images)
    num_rows = (num_images - 1) // 3 + 1  # Calculate rows needed for original images
    total_rows = num_rows * 2  # Double the rows to accommodate reconstructed images

    plt.figure(figsize=(15, 10))
    plt.suptitle("Original vs Rotated Images")

    for i in range(num_images):
        plt.subplot(total_rows, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Original {i+1}")
        plt.axis("off")

    start_idx = num_rows * 3  # Start index for rotated images
    for i in range(num_images):
        plt.subplot(total_rows, 3, start_idx + i + 1)
        plt.imshow(rotated_images[i])
        plt.title(f"Rotated {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def vertical_flip_image(images: Union[list[np.ndarray], list[Image.Image]]) -> list[np.ndarray]:
    transform = A.VerticalFlip(p=1.0)

    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        transformed_image = transform(image=image)["image"]
        transformed_images.append(transformed_image)

    return transformed_images


def horizontal_flip_image(images: Union[list[np.ndarray], list[Image.Image]]) -> list[np.ndarray]:
    transform = A.HorizontalFlip(p=1.0)

    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        transformed_image = transform(image=image)["image"]
        transformed_images.append(transformed_image)

    return transformed_images


def crop_image():
    pass


def resize_image():
    pass


def shear_image():
    pass


def brightness_image():
    pass


def contrast_image():
    pass


def saturation_image():
    pass


def hue_image():
    pass


def color_jitter_image():
    pass


def multiple_transformations():
    pass


def normalize_pixel_values():
    pass
