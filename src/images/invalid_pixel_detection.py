from typing import Union

from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from src.images.utils import load_image


def detect_missing_data(img_json: dict) -> dict:
    """
    Detect missing data in images and add missing coordinates to the json.
    
    Args:
        img_json: Dictionary containing image paths and path_to_id mapping
        
    Returns:
        Updated img_json with missing_coords field added
    """
    missing_coords_dict = {}
    
    for img_path in img_json["image_paths"]:
        image = load_image(img_path)

        # Get coordinates of missing pixels (all channels are 0)
        missing_mask = np.all(image == 0, axis=2)
        rows, cols = np.where(missing_mask)
        missing_coords = np.column_stack((rows, cols))
        
        # Store coords using image id as key
        img_id = img_json["path_to_id"][img_path]
        missing_coords_dict[img_id] = missing_coords.tolist()

    img_json["missing_coords"] = missing_coords_dict
    
    return img_json


def detect_pixel_level_outliers(
    img_json: dict,
    method: str = "lof",
    *,
    n_neighbors: int = 20,
    contamination: float = 0.001,
) -> dict:
    """
    Detect outlier pixels in images and add outlier coordinates to the json.

    Args:
        img_json: Dictionary containing image paths and path_to_id mapping
        method: Outlier detection method (currently only supports "lof")
        n_neighbors: Number of neighbors for LOF algorithm
        contamination: Expected proportion of outliers in the dataset

    Returns:
        Updated img_json with outlier_coords field added
    """
    if method not in ["lof"]:
        raise NotImplementedError(f"Invalid method: {method}")

    outlier_coords_dict = {}

    for img_path in img_json["image_paths"]:
        image = load_image(img_path)

        # Reshape image to 2D array (n_pixels, n_channels)
        image_2d = image.reshape(-1, image.shape[-1] if len(image.shape) > 2 else 1)

        if method == "lof":
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            outlier_labels = lof.fit_predict(image_2d)
            # Reshape back to original image shape
            outlier_labels = outlier_labels.reshape(image.shape[:-1])

            # Get coordinates of outlier pixels
            rows, cols = np.where(outlier_labels == -1)
            outlier_coords = np.column_stack((rows, cols))

            # Store coords using image id as key
            img_id = img_json["path_to_id"][img_path]
            outlier_coords_dict[img_id] = outlier_coords.tolist()

    img_json["outlier_coords"] = outlier_coords_dict

    return img_json


# def visualize_missing_data(
#     image: Union[np.ndarray, Image.Image],
# ) -> None:
#     if not isinstance(image, np.ndarray):
#         image = np.array(image)

#     missing_coords = detect_missing_data(image)

#     plt.figure(figsize=(10, 8))
#     plt.imshow(image)
#     plt.title("Missing Data")
#     plt.scatter(
#         missing_coords[:, 1], missing_coords[:, 0], color="red", marker="o", s=100
#     )
#     plt.axis("off")
#     plt.show()

# def visualize_imputed_data(
#     image: Union[np.ndarray, Image.Image],
#     mode: str = "mean",
# ) -> None:
#     if not isinstance(image, np.ndarray):
#         image = np.array(image)

#     imputed_image = impute_missing_data(image, mode)

#     plt.figure(figsize=(12, 6))
#     plt.suptitle(f"Imputation using {mode} method")

#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.title("Original Image")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(imputed_image)
#     plt.title("Imputed Image")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()


# def visualize_interpolated_data(
#     image: Union[np.ndarray, Image.Image], method: str = "linear"
# ):
#     if not isinstance(image, np.ndarray):
#         image = np.array(image)

#     interpolated_image = interpolate_missing_data(image, method)

#     plt.figure(figsize=(12, 6))
#     plt.suptitle(f"Interpolation using {method} method")

#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.title("Original Image")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(interpolated_image)
#     plt.title("Interpolated Image")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()
