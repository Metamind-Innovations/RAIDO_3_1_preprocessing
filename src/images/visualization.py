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
