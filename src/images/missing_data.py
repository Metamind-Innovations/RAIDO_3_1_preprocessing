from typing import Union

import numpy as np
from PIL import Image
from scipy import interpolate


def detect_missing_data(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Detect missing data in an image"""
    # Convert PIL image to numpy array if necessary
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Get coordinates of missing pixels (all channels are 0)
    missing_mask = np.all(image == 0, axis=2)
    rows, cols = np.where(missing_mask)
    missing_coords = np.column_stack((rows, cols))

    return missing_coords


def impute_missing_data(
    image: Union[np.ndarray, Image.Image], coords: np.ndarray, mode: str = "mean"
) -> np.ndarray:
    """Impute missing data in an image using a single value"""

    # Convert PIL image to numpy array if necessary
    if not isinstance(image, np.ndarray):
        image = np.array(image)

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
    image: Union[np.ndarray, Image.Image], coords: np.ndarray, method: str = "linear"
) -> np.ndarray:
    """Interpolate missing data in an image using a specified method"""

    # Ensure method is valid
    if method not in ["linear", "nearest", "cubic"]:
        raise NotImplementedError(f"Invalid method: {method}")

    # Convert PIL image to numpy array if necessary
    if not isinstance(image, np.ndarray):
        image = np.array(image)

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
