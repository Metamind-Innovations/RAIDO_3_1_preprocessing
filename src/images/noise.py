from typing import Union

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from skimage import img_as_float
from skimage.restoration import denoise_nl_means


def denoise_non_local_means(
    image: Union[np.ndarray, Image.Image],
    patch_size: int = 7,
    patch_distance: int = 11,
    fast_mode: bool = False,
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = img_as_float(image)

    # Determine channel_axis based on image dimensions
    channel_axis = -1 if len(image.shape) > 2 else None

    patch_kw = dict(
        patch_size=patch_size, patch_distance=patch_distance, channel_axis=channel_axis
    )

    denoised_image = denoise_nl_means(image, fast_mode=fast_mode, **patch_kw)

    noise_mask = image - denoised_image

    return denoised_image, noise_mask


def visualize_denoised_image(
    image: Union[np.ndarray, Image.Image],
    patch_size: int = 7,
    patch_distance: int = 11,
    fast_mode: bool = False,
) -> None:
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    denoised_image, noise_mask = denoise_non_local_means(
        image, patch_size, patch_distance, fast_mode,
    )

    # Normalize noise mask to [0,1] range
    noise_mask = (noise_mask - noise_mask.min()) / (noise_mask.max() - noise_mask.min())

    plt.figure(figsize=(12, 8))
    plt.suptitle("Non-Local Means Denoising")

    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)

    ax1 = plt.subplot(gs[0, :2])
    ax2 = plt.subplot(gs[0, 2:])
    ax3 = plt.subplot(gs[1, 1:3])

    # Plot original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Plot denoised image
    ax2.imshow(denoised_image)
    ax2.set_title("Denoised Image")
    ax2.axis("off")

    # Plot noise mask
    ax3.imshow(noise_mask)
    ax3.set_title("Noise Mask")
    ax3.axis("off")

    plt.show()
