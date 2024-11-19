from typing import Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def detect_image_level_outliers(
    images_input: np.ndarray,
    method: str = "isolation_forest",
    *,
    n_estimators: int = 100,
    contamination: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Detect outlier images using unsupervised outlier detection algorithms.

    Args:
        images_input: Array of images with shape (n_images, height, width, channels)
        method: Outlier detection method. Currently supported methods:
            - "isolation_forest": Uses Isolation Forest algorithm
        n_estimators: Number of base estimators in the ensemble for Isolation Forest
        contamination: Expected proportion of outliers in the dataset, between 0 and 0.5
        random_state: Random state for reproducible results

    Returns:
        np.ndarray: Array of outlier labels with shape (n_images,), where:
            -1 indicates outliers
            1 indicates inliers
    """

    if method not in ["isolation_forest"]:
        raise NotImplementedError(f"Invalid method: {method}")

    # Reshape images to 2D array (n_samples, n_features)
    n_samples = images_input.shape[0]
    flattened_images = images_input.reshape(n_samples, -1)

    if method == "isolation_forest":
        iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        outlier_labels = iso_forest.fit_predict(flattened_images)

    return outlier_labels


def detect_pixel_level_outliers(
    image: Union[np.ndarray, Image.Image],
    method: str = "lof",
    *,
    n_neighbors: int = 20,
    contamination: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outlier pixels using Local Outlier Factor algorithm.

    Args:
        image: Image to detect outliers in
        method: Outlier detection method (currently only supports "lof")

    Returns:
        Tuple of masked image and outlier labels
    """

    if method not in ["lof"]:
        raise NotImplementedError(f"Invalid method: {method}")

    # Convert PIL image to numpy array if necessary
    if not isinstance(image, np.ndarray):
        image = np.array(image)

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

        masked_image = image.copy()
        masked_image[outlier_labels == -1] = 0

    return outlier_coords


def visualize_image_outliers(
    images_input: np.ndarray,
    method: str = "isolation_forest",
    *,
    n_estimators: int = 100,
    contamination: float = 0.1,
    random_state: int = 42,
):
    outlier_labels = detect_image_level_outliers(
        images_input,
        method,
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )

    num_images = len(images_input)
    num_rows = (num_images - 1) // 3 + 1  # Calculate number of rows needed

    plt.figure(figsize=(12, 4 * num_rows))
    plt.suptitle(f"Outlier Images detection using {method}")
    for i, image in enumerate(images_input):
        ax = plt.subplot(num_rows, min(3, num_images), i + 1)
        plt.imshow(image)

        # Add red rectangle around outlier images
        if outlier_labels[i] == -1:  # -1 indicates outlier
            # Create a red rectangle patch with small margin from borders
            margin = 5  # pixels from border
            rect = plt.Rectangle(
                (0, margin),  # (x,y) of lower left corner
                image.shape[1] - margin,  # width
                image.shape[0] - 2 * margin,  # height
                fill=False,
                edgecolor="red",
                linewidth=5,
            )
            ax.add_patch(rect)
            plt.title("Outlier")
        else:
            plt.title("Normal")

        plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_pixel_outliers(
    image: Union[np.ndarray, Image.Image],
    method: str = "lof",
    *,
    n_neighbors: int = 20,
    contamination: float = 0.1,
):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    outlier_coords = detect_pixel_level_outliers(
        image, method, n_neighbors=n_neighbors, contamination=contamination
    )

    plt.figure(figsize=(10, 8))
    plt.title(f"Outlier pixels detection using {method}")
    plt.imshow(image)
    plt.scatter(
        outlier_coords[:, 1],
        outlier_coords[:, 0],
        color="red",
        marker="o",
        s=20,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()
