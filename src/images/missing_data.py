from typing import Union

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy import interpolate


def detect_missing_data(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Detect missing data in an image"""

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Get coordinates of missing pixels (all channels are 0)
    missing_mask = np.all(image == 0, axis=2)
    rows, cols = np.where(missing_mask)
    missing_coords = np.column_stack((rows, cols))

    return missing_coords


def impute_missing_data(
    image: Union[np.ndarray, Image.Image], mode: str = "mean"
) -> np.ndarray:
    """Impute missing data in an image using a single value"""

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    coords = detect_missing_data(image)

    missing_pixels_mask = np.zeros(image.shape[:2], dtype=bool)
    missing_pixels_mask[coords[:, 0], coords[:, 1]] = True

    imputed_image = image.copy()

    if mode == "mean":
        imputation_value = np.mean(image[~missing_pixels_mask])
    elif mode == "median":
        imputation_value = np.median(image[~missing_pixels_mask])
    else:
        raise NotImplementedError(f"Invalid mode: {mode}")

    imputed_image[missing_pixels_mask] = imputation_value

    return imputed_image


def interpolate_missing_data(
    image: Union[np.ndarray, Image.Image], method: str = "linear"
) -> np.ndarray:
    """Interpolate missing data in an image using a specified method"""

    # Ensure method is valid
    if method not in ["linear", "nearest", "cubic"]:
        raise NotImplementedError(f"Invalid method: {method}")

    # Convert PIL image to numpy array if necessary
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    coords = detect_missing_data(image)

    # Coordinates for all pixels
    rows, cols = np.indices(image.shape[:2])

    # Coordinates of valid pixels
    missing_pixels_mask = np.zeros(image.shape[:2], dtype=bool)
    missing_pixels_mask[coords[:, 0], coords[:, 1]] = True
    valid_mask = ~missing_pixels_mask
    valid_coords = np.column_stack((rows[valid_mask], cols[valid_mask]))
    valid_values = image[valid_mask]

    # Create interpolator for each channel
    interpolated_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        channel_interpolator = interpolate.griddata(
            valid_coords,
            valid_values[:, channel],
            np.column_stack((rows.ravel(), cols.ravel())),
            method=method,
        )
        interpolated_image[:, :, channel] = channel_interpolator.reshape(
            image.shape[:2]
        )

    return interpolated_image


def visualize_missing_data(
    image: Union[np.ndarray, Image.Image],
) -> None:
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    missing_coords = detect_missing_data(image)

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title("Missing Data")
    plt.scatter(
        missing_coords[:, 1], missing_coords[:, 0], color="red", marker="o", s=100
    )
    plt.axis("off")
    plt.show()


def visualize_imputed_data(
    image: Union[np.ndarray, Image.Image],
    mode: str = "mean",
) -> None:
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    imputed_image = impute_missing_data(image, mode)

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Imputation using {mode} method")

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Plot imputed image
    plt.subplot(1, 2, 2)
    plt.imshow(imputed_image)
    plt.title("Imputed Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_interpolated_data(
    image: Union[np.ndarray, Image.Image],
    method: str = "linear"
):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    interpolated_image = interpolate_missing_data(image, method)

    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Interpolation using {method} method")

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Plot interpoalted image
    plt.subplot(1, 2, 2)
    plt.imshow(interpolated_image)
    plt.title("Interpolated Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
