from typing import Union

import albumentations as A
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


def transform_images(
    images: Union[list[np.ndarray], list[Image.Image]],
    transformation_list: list[str],
    **transform_kwargs,
) -> list[np.ndarray]:
    transformed_images = images.copy()

    # Define valid parameters for each transformation
    valid_params = {
        "rotation": ["negative_angle", "positive_angle"],
        "crop": ["min_crop_height", "max_crop_height", "w2h_ratio"],
        "resize": ["height", "width"],
        "shear": ["min_shear_angle", "max_shear_angle"],
        "brightness": ["brightness_min_factor", "brightness_max_factor"],
        "contrast": ["contrast_min_factor", "contrast_max_factor"],
        "saturation": ["saturation_min_factor", "saturation_max_factor"],
        "hue": ["hue_min_factor", "hue_max_factor"],
    }

    for transformation in transformation_list:
        if transformation not in valid_params and transformation not in [
            "vertical_flip",
            "horizontal_flip",
        ]:
            raise ValueError(f"Transformation {transformation} not found")

        # Filter kwargs to only include valid parameters for this transformation
        filtered_kwargs = {}
        if transformation in valid_params:
            filtered_kwargs = {
                k: v
                for k, v in transform_kwargs.items()
                if k in valid_params[transformation]
            }

        if transformation == "rotation":
            transformed_images = rotate_image(transformed_images, **filtered_kwargs)
        elif transformation == "vertical_flip":
            transformed_images = vertical_flip_image(transformed_images)
        elif transformation == "horizontal_flip":
            transformed_images = horizontal_flip_image(transformed_images)
        elif transformation == "crop":
            transformed_images = crop_image(transformed_images, **transform_kwargs)
        elif transformation == "resize":
            transformed_images = resize_image(transformed_images, **transform_kwargs)
        elif transformation == "shear":
            transformed_images = shear_image(transformed_images, **transform_kwargs)
        elif transformation == "brightness":
            transformed_images = brightness_image(
                transformed_images, **transform_kwargs
            )
        elif transformation == "contrast":
            transformed_images = contrast_image(transformed_images, **transform_kwargs)
        elif transformation == "saturation":
            transformed_images = saturation_image(
                transformed_images, **transform_kwargs
            )
        elif transformation == "hue":
            transformed_images = hue_image(transformed_images, **transform_kwargs)

    return transformed_images


def visualize_transformed_images(
    images: Union[list[np.ndarray], list[Image.Image]],
    transformation_list: list[str],
    **transform_kwargs,
) -> None:
    transformed_images = transform_images(
        images, transformation_list, **transform_kwargs
    )

    num_images = len(images)
    num_rows = (num_images - 1) // 3 + 1  # Calculate rows needed for original images
    total_rows = num_rows * 2  # Double the rows to accommodate transformed images

    plt.figure(figsize=(15, 10))
    plt.suptitle("Original vs Transformed Images")

    for i in range(num_images):
        plt.subplot(total_rows, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Original {i+1}")
        plt.axis("off")

    start_idx = num_rows * 3  # Start index for transformed images
    for i in range(num_images):
        plt.subplot(total_rows, 3, start_idx + i + 1)
        plt.imshow(transformed_images[i])
        plt.title(f"Transformed {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# TODO: Maybe in next version add the rest of the parameters for Rotate
def rotate_image(
    images: Union[list[np.ndarray], list[Image.Image]],
    negative_angle: int = -90,
    positive_angle: int = 90,
) -> list[np.ndarray]:
    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        transform = A.Rotate(limit=(negative_angle, positive_angle), p=1.0)

        transformed_image = transform(image=image)["image"]
        transformed_images.append(transformed_image)

    return transformed_images


def vertical_flip_image(
    images: Union[list[np.ndarray], list[Image.Image]],
) -> list[np.ndarray]:
    transform = A.VerticalFlip(p=1.0)

    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        transformed_image = transform(image=image)["image"]
        transformed_images.append(transformed_image)

    return transformed_images


def horizontal_flip_image(
    images: Union[list[np.ndarray], list[Image.Image]],
) -> list[np.ndarray]:
    transform = A.HorizontalFlip(p=1.0)

    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        transformed_image = transform(image=image)["image"]
        transformed_images.append(transformed_image)

    return transformed_images


def crop_image(
    images: Union[list[np.ndarray], list[Image.Image]],
    min_crop_height: int = 180,
    max_crop_height: int = 320,
    w2h_ratio: float = 1.0,
):
    """
    Crop image and resize to the original size.
    """
    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        height, width = image.shape[:2]

        transform = A.RandomSizedCrop(
            min_max_height=(min_crop_height, max_crop_height),
            size=(height, width),
            w2h_ratio=w2h_ratio,
        )

        transformed_image = transform(image=image)["image"]
        transformed_images.append(transformed_image)

    return transformed_images


def resize_image(
    images: Union[list[np.ndarray], list[Image.Image]],
    height: int = 320,
    width: int = 180,
):
    """
    Resize image to the given height and width.
    """
    transformed_images = []

    transform = A.Resize(height=height, width=width)

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        transformed_image = transform(image=image)["image"]
        transformed_images.append(transformed_image)

    return transformed_images


def shear_image(
    images: Union[list[np.ndarray], list[Image.Image]],
    min_shear_angle: int = -45,
    max_shear_angle: int = 45,
):
    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        transform = A.Affine(shear=(min_shear_angle, max_shear_angle), p=1.0)

        transformed_image = transform(image=image)["image"]
        transformed_images.append(transformed_image)

    return transformed_images


def brightness_image(
    images: Union[list[np.ndarray], list[Image.Image]],
    brightness_min_factor: float = 0.2,
    brightness_max_factor: float = 2.0,
):
    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Handle 4-channel (RGBA) images
        if image.shape[-1] == 4:
            # Split alpha channel
            rgb = image[..., :3]
            alpha = image[..., 3]

            # Apply transformation to RGB channels
            transform = A.ColorJitter(
                brightness=(brightness_min_factor, brightness_max_factor),
                contrast=(1.0, 1.0),
                saturation=(1.0, 1.0),
                hue=(0.0, 0.0),
                p=1.0,
            )
            transformed_rgb = transform(image=rgb)["image"]

            # Recombine with alpha channel
            transformed_image = np.dstack((transformed_rgb, alpha))
        else:
            # Original behavior for 1 or 3 channel images
            transform = A.ColorJitter(
                brightness=(brightness_min_factor, brightness_max_factor),
                contrast=(1.0, 1.0),
                saturation=(1.0, 1.0),
                hue=(0.0, 0.0),
                p=1.0,
            )
            transformed_image = transform(image=image)["image"]

        transformed_images.append(transformed_image)

    return transformed_images


def contrast_image(
    images: Union[list[np.ndarray], list[Image.Image]],
    contrast_min_factor: float = 0.2,
    contrast_max_factor: float = 2.0,
):
    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Handle 4-channel (RGBA) images
        if image.shape[-1] == 4:
            # Split alpha channel
            rgb = image[..., :3]
            alpha = image[..., 3]

            # Apply transformation to RGB channels
            transform = A.ColorJitter(
                brightness=(1.0, 1.0),
                contrast=(contrast_min_factor, contrast_max_factor),
                saturation=(1.0, 1.0),
                hue=(0.0, 0.0),
                p=1.0,
            )
            transformed_rgb = transform(image=rgb)["image"]

            # Recombine with alpha channel
            transformed_image = np.dstack((transformed_rgb, alpha))
        else:
            # Original behavior for 1 or 3 channel images
            transform = A.ColorJitter(
                brightness=(1.0, 1.0),
                contrast=(contrast_min_factor, contrast_max_factor),
                saturation=(1.0, 1.0),
                hue=(0.0, 0.0),
                p=1.0,
            )
            transformed_image = transform(image=image)["image"]

        transformed_images.append(transformed_image)

    return transformed_images


def saturation_image(
    images: Union[list[np.ndarray], list[Image.Image]],
    saturation_min_factor: float = 0.2,
    saturation_max_factor: float = 2.0,
):
    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Handle 4-channel (RGBA) images
        if image.shape[-1] == 4:
            # Split alpha channel
            rgb = image[..., :3]
            alpha = image[..., 3]

            # Apply transformation to RGB channels
            transform = A.ColorJitter(
                brightness=(1.0, 1.0),
                contrast=(1.0, 1.0),
                saturation=(saturation_min_factor, saturation_max_factor),
                hue=(0.0, 0.0),
                p=1.0,
            )
            transformed_rgb = transform(image=rgb)["image"]

            # Recombine with alpha channel
            transformed_image = np.dstack((transformed_rgb, alpha))
        else:
            # Original behavior for 1 or 3 channel images
            transform = A.ColorJitter(
                brightness=(1.0, 1.0),
                contrast=(1.0, 1.0),
                saturation=(saturation_min_factor, saturation_max_factor),
                hue=(0.0, 0.0),
                p=1.0,
            )
            transformed_image = transform(image=image)["image"]

        transformed_images.append(transformed_image)

    return transformed_images


def hue_image(
    images: Union[list[np.ndarray], list[Image.Image]],
    hue_min_factor: float = -0.5,
    hue_max_factor: float = 0.5,
):
    transformed_images = []

    for image in images:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Handle 4-channel (RGBA) images
        if image.shape[-1] == 4:
            # Split alpha channel
            rgb = image[..., :3]
            alpha = image[..., 3]

            # Apply transformation to RGB channels
            transform = A.ColorJitter(
                brightness=(1.0, 1.0),
                contrast=(1.0, 1.0),
                saturation=(1.0, 1.0),
                hue=(hue_min_factor, hue_max_factor),
                p=1.0,
            )
            transformed_rgb = transform(image=rgb)["image"]

            # Recombine with alpha channel
            transformed_image = np.dstack((transformed_rgb, alpha))
        else:
            # Original behavior for 1 or 3 channel images
            transform = A.ColorJitter(
                brightness=(1.0, 1.0),
                contrast=(1.0, 1.0),
                saturation=(1.0, 1.0),
                hue=(hue_min_factor, hue_max_factor),
                p=1.0,
            )
            transformed_image = transform(image=image)["image"]

        transformed_images.append(transformed_image)

    return transformed_images


def normalize_pixel_values(
    image: Union[np.ndarray, Image.Image],
):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    normalized_image = (image - image.min()) / (image.max() - image.min())

    return normalized_image
