from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_image(image: Union[np.ndarray, Image.Image], title: str = None):
    """Plot an image with an optional title"""
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def plot_multiple_images(
    images: Union[np.ndarray, List[Image.Image]],
    titles: List[str] = None,
):
    """Plot multiple images with optional titles"""
    num_images = len(images)
    num_rows = (num_images - 1) // 3 + 1

    plt.figure(figsize=(12, 4 * num_rows))
    for i, image in enumerate(images):
        plt.subplot(num_rows, min(3, num_images), i + 1)
        plt.imshow(image)
        if titles:
            plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# def plot_missing_pixels(
#     image: Union[np.ndarray, Image.Image],
#     missing_coords: np.ndarray,
#     title: str = None,
# ) -> None:
#     """Plot missing data in an image with an optional title"""
#     # Convert PIL image to numpy array if needed
#     if not isinstance(image, np.ndarray):
#         image = np.array(image)

#     # Plot the original image
#     plt.imshow(image)

#     # Plot red 'o' markers for missing data
#     plt.scatter(
#         missing_coords[:, 1], missing_coords[:, 0], color="red", marker="o", s=100
#     )
#     if title:
#         plt.title(title)
#     plt.axis("off")
#     plt.show()


def plot_image_outliers(
    images: Union[np.ndarray, List[Image.Image]],
    outlier_labels: np.ndarray,
):
    """Plot multiple images with outlier detection results"""
    num_images = len(images)
    num_rows = (num_images - 1) // 3 + 1  # Calculate number of rows needed

    plt.figure(figsize=(12, 4 * num_rows))
    for i, image in enumerate(images):
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

        plt.axis("off")
    plt.tight_layout()
    plt.show()
