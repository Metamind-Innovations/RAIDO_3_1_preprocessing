import numpy as np
from PIL import Image
from scipy import interpolate

from src.images.utils import load_image


def impute_invalid_pixels(img_json: dict, mode: str = "mean") -> dict:
    """
    Impute missing and outlier pixels in images using a single value and save imputed images.

    Args:
        img_json: Dictionary containing image paths, path_to_id mapping, missing_coords and outlier_coords
        mode: Imputation mode - either 'mean' or 'median'

    Returns:
        The input img_json dict unchanged
    """
    if mode not in ["mean", "median"]:
        raise NotImplementedError(f"Invalid mode: {mode}")

    for img_path in img_json["image_paths"]:
        image = load_image(img_path)

        img_id = img_json["path_to_id"][img_path]
        missing_coords = np.array(img_json["missing_coords"][img_id])
        outlier_coords = np.array(img_json["outlier_coords"][img_id])

        # Create masks for both missing and outlier pixels
        invalid_pixels_mask = np.zeros(image.shape[:2], dtype=bool)
        if len(missing_coords) > 0:
            invalid_pixels_mask[missing_coords[:, 0], missing_coords[:, 1]] = True
        if len(outlier_coords) > 0:
            invalid_pixels_mask[outlier_coords[:, 0], outlier_coords[:, 1]] = True

        # Only process if there are any invalid pixels
        if np.any(invalid_pixels_mask):
            imputed_image = image.copy()

            if mode == "mean":
                imputation_value = np.mean(image[~invalid_pixels_mask])
            elif mode == "median":
                imputation_value = np.median(image[~invalid_pixels_mask])
            else:
                raise NotImplementedError(f"Invalid mode: {mode}")

            imputed_image[invalid_pixels_mask] = imputation_value

            # Save imputed image back to original path
            Image.fromarray(imputed_image.astype(np.uint8)).save(img_path)

    return img_json


def interpolate_invalid_pixels(img_json: dict, method: str = "linear") -> dict:
    """
    Interpolate missing and outlier pixels in images using spatial interpolation and save interpolated images.

    Args:
        img_json: Dictionary containing image paths, path_to_id mapping, missing_coords and outlier_coords
        method: Interpolation method - either 'linear', 'nearest', or 'cubic'

    Returns:
        The input img_json dict unchanged
    """
    if method not in ["linear", "nearest", "cubic"]:
        raise NotImplementedError(f"Invalid method: {method}")

    for img_path in img_json["image_paths"]:
        image = load_image(img_path)

        img_id = img_json["path_to_id"][img_path]
        missing_coords = np.array(img_json["missing_coords"][img_id])
        outlier_coords = np.array(img_json["outlier_coords"][img_id])

        # Create masks for both missing and outlier pixels
        invalid_pixels_mask = np.zeros(image.shape[:2], dtype=bool)
        if len(missing_coords) > 0:
            invalid_pixels_mask[missing_coords[:, 0], missing_coords[:, 1]] = True
        if len(outlier_coords) > 0:
            invalid_pixels_mask[outlier_coords[:, 0], outlier_coords[:, 1]] = True

        # Only process if there are any invalid pixels
        if np.any(invalid_pixels_mask):
            # Coordinates for all pixels
            rows, cols = np.indices(image.shape[:2])

            # Get valid pixel coordinates and values
            valid_mask = ~invalid_pixels_mask
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

            # Save interpolated image back to original path
            Image.fromarray(interpolated_image.astype(np.uint8)).save(img_path)

    return img_json
