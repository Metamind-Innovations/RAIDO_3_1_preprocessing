import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

from src.images.utils import load_image


def transform_images(
    img_json: dict,
    transformation_list: list[str],
    **transform_kwargs,
) -> dict:
    """
    Apply image transformations in a specific order to all images in img_json.

    Args:
        img_json: Dictionary containing image paths and path_to_id mapping
        transformation_list: List of transformations to apply
        transform_kwargs: Transformation-specific parameters

    Returns:
        Updated img_json
    """
    # Valid parameters and order for each transformation
    valid_params = {
        "crop": ["min_crop_height", "max_crop_height", "w2h_ratio"],
        "resize": ["height", "width"],
        "rotation": ["negative_angle", "positive_angle"],
        "shear": ["min_shear_angle", "max_shear_angle"],
        "horizontal_flip": [],
        "vertical_flip": [],
        "brightness": ["brightness_min_factor", "brightness_max_factor"],
        "contrast": ["contrast_min_factor", "contrast_max_factor"],
        "saturation": ["saturation_min_factor", "saturation_max_factor"],
        "hue": ["hue_min_factor", "hue_max_factor"],
    }

    # Transformation order
    transform_order = [
        "crop",
        "resize",
        "rotation",
        "shear",
        "horizontal_flip",
        "vertical_flip",
        "brightness",
        "contrast",
        "saturation",
        "hue",
    ]

    # Validate transformations
    for transform in transformation_list:
        if transform not in valid_params:
            raise ValueError(f"Transformation {transform} not found")

    # Sort transformations according to predefined order
    ordered_transformations = [t for t in transform_order if t in transformation_list]

    # Store original paths to iterate over
    original_paths = img_json["image_paths"].copy()

    # Clear image_paths and rebuild it with both original and augmented images
    img_json["image_paths"] = []

    for img_path in original_paths:
        # Add original image path back
        img_json["image_paths"].append(img_path)

        image = load_image(img_path)
        transformed_image = np.array(image)

        for transform in ordered_transformations:
            # Filter kwargs for current transformation
            filtered_kwargs = {
                k: v
                for k, v in transform_kwargs.items()
                if k in valid_params[transform]
            }

            if transform == "crop":
                transformed_image = crop_image(transformed_image, **filtered_kwargs)
            elif transform == "resize":
                transformed_image = resize_image(transformed_image, **filtered_kwargs)
            elif transform == "rotation":
                transformed_image = rotate_image(transformed_image, **filtered_kwargs)
            elif transform == "shear":
                transformed_image = shear_image(transformed_image, **filtered_kwargs)
            elif transform == "horizontal_flip":
                transformed_image = horizontal_flip_image(transformed_image)
            elif transform == "vertical_flip":
                transformed_image = vertical_flip_image(transformed_image)
            elif transform == "brightness":
                transformed_image = brightness_image(
                    transformed_image, **filtered_kwargs
                )
            elif transform == "contrast":
                transformed_image = contrast_image(transformed_image, **filtered_kwargs)
            elif transform == "saturation":
                transformed_image = saturation_image(
                    transformed_image, **filtered_kwargs
                )
            elif transform == "hue":
                transformed_image = hue_image(transformed_image, **filtered_kwargs)

        # Augmented image path
        base_path = img_path.rsplit(".", 1)[0]
        ext = img_path.rsplit(".", 1)[1]
        augmented_path = f"{base_path}_augmented.{ext}"

        # Save transformed image
        Image.fromarray(transformed_image.astype(np.uint8)).save(augmented_path)

        # Add new augmented path to image_paths
        img_json["image_paths"].append(augmented_path)

        # Get original image ID and create new ID for augmented image
        original_id = img_json["path_to_id"][img_path]
        augmented_id = f"{original_id}_augmented"

        # Update path_to_id mapping
        img_json["path_to_id"][augmented_path] = augmented_id

        # Update any existing labels
        if "labels" in img_json:
            if original_id in img_json["labels"]:
                img_json["labels"][augmented_id] = img_json["labels"][original_id]

        # Update any other ID-based fields
        for field in img_json:
            if isinstance(img_json[field], dict) and original_id in img_json[field]:
                if field != "path_to_id":
                    img_json[field][augmented_id] = img_json[field][original_id]

    # Update labels CSV
    if "label_path" in img_json:
        df = pd.read_csv(img_json["label_path"], index_col=False)

        # Create new rows for augmented images
        new_rows = []
        for img_path in original_paths:
            # Get the filename without path and extension to match CSV format
            original_id = img_path.split("/")[-1].rsplit(".", 1)[0]
            augmented_id = f"{original_id}_augmented"

            # Find original row and create new row with augmented data
            original_row = df[df["IMAGE"] == original_id]
            if not original_row.empty:
                new_row = original_row.iloc[0].copy()
                new_row["IMAGE"] = augmented_id
                new_rows.append(new_row)

        if new_rows:  # Only concat if we have new rows to add
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df.to_csv(img_json["label_path"], index=False)

    return img_json


# TODO: Maybe in next version add the rest of the parameters for Rotate
def rotate_image(
    image: np.ndarray,
    negative_angle: int = -90,
    positive_angle: int = 90,
) -> np.ndarray:
    transform = A.Rotate(limit=(negative_angle, positive_angle), p=1.0)
    transformed_image = transform(image=image)["image"]

    return transformed_image


def vertical_flip_image(
    image: np.ndarray,
) -> np.ndarray:
    transform = A.VerticalFlip(p=1.0)
    transformed_image = transform(image=image)["image"]

    return transformed_image


def horizontal_flip_image(
    image: np.ndarray,
) -> np.ndarray:
    transform = A.HorizontalFlip(p=1.0)
    transformed_image = transform(image=image)["image"]

    return transformed_image


def crop_image(
    image: np.ndarray,
    min_crop_height: int = 180,
    max_crop_height: int = 320,
    w2h_ratio: float = 1.0,
) -> np.ndarray:
    height, width = image.shape[:2]

    transform = A.RandomSizedCrop(
        min_max_height=(min_crop_height, max_crop_height),
        size=(height, width),
        w2h_ratio=w2h_ratio,
    )
    transformed_image = transform(image=image)["image"]

    return transformed_image


def resize_image(
    image: np.ndarray,
    height: int = 320,
    width: int = 180,
) -> np.ndarray:
    transform = A.Resize(height=height, width=width)
    transformed_image = transform(image=image)["image"]

    return transformed_image


def shear_image(
    image: np.ndarray,
    min_shear_angle: int = -45,
    max_shear_angle: int = 45,
) -> np.ndarray:
    transform = A.Affine(shear=(min_shear_angle, max_shear_angle), p=1.0)
    transformed_image = transform(image=image)["image"]

    return transformed_image


def brightness_image(
    image: np.ndarray,
    brightness_min_factor: float = 0.2,
    brightness_max_factor: float = 2.0,
) -> np.ndarray:
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

    return transformed_image


def contrast_image(
    image: np.ndarray,
    contrast_min_factor: float = 0.2,
    contrast_max_factor: float = 2.0,
) -> np.ndarray:
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

    return transformed_image


def saturation_image(
    image: np.ndarray,
    saturation_min_factor: float = 0.2,
    saturation_max_factor: float = 2.0,
) -> np.ndarray:
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

    return transformed_image


def hue_image(
    image: np.ndarray,
    hue_min_factor: float = -0.5,
    hue_max_factor: float = 0.5,
) -> np.ndarray:
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

    return transformed_image


# def visualize_transformed_images(
#     images: Union[list[np.ndarray], list[Image.Image]],
#     transformation_list: list[str],
#     **transform_kwargs,
# ) -> None:
#     transformed_images = transform_images(
#         images, transformation_list, **transform_kwargs
#     )

#     num_images = len(images)
#     num_rows = (num_images - 1) // 3 + 1  # Calculate rows needed for original images
#     total_rows = num_rows * 2  # Double the rows to accommodate transformed images

#     plt.figure(figsize=(15, 10))
#     plt.suptitle("Original vs Transformed Images")

#     for i in range(num_images):
#         plt.subplot(total_rows, 3, i + 1)
#         plt.imshow(images[i])
#         plt.title(f"Original {i+1}")
#         plt.axis("off")

#     start_idx = num_rows * 3  # Start index for transformed images
#     for i in range(num_images):
#         plt.subplot(total_rows, 3, start_idx + i + 1)
#         plt.imshow(transformed_images[i])
#         plt.title(f"Transformed {i+1}")
#         plt.axis("off")

#     plt.tight_layout()
#     plt.show()
