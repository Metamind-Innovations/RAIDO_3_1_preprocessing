from typing import Union

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage import img_as_float
from skimage.restoration import denoise_nl_means


def denoise_non_local_means(
    image: Union[np.ndarray, Image.Image],
    patch_size: int = 7,
    patch_distance: int = 11,
    fast_mode: bool = False,
    multichannel: bool = False,
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = img_as_float(image)

    if multichannel:
        channel_axis = -1
    else:
        channel_axis = None

    patch_kw = dict(
        patch_size=patch_size, patch_distance=patch_distance, channel_axis=channel_axis
    )

    denoised_image = denoise_nl_means(image, fast_mode=fast_mode, **patch_kw)

    return denoised_image


def visualize_denoised_image(
    image: Union[np.ndarray, Image.Image],
    patch_size: int = 7,
    patch_distance: int = 11,
    fast_mode: bool = False,
    multichannel: bool = False,
) -> None:
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    denoised_image = denoise_non_local_means(
        image, patch_size, patch_distance, fast_mode, multichannel
    )

    plt.figure(figsize=(12, 6))
    plt.suptitle("Non-Local Means Denoising")

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Plot denoised image
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image)
    plt.title("Denoised Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# TODO: Return or create a function that returns the noise mask
