from typing import Union

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


def pca_single_image(
    image: Union[np.ndarray, Image.Image],
    n_components: int = 30,
):
    """
    reconstructed_image is normalized to [0, 1]
    pca_image is not normalized, has the values that occur after applying PCA meaning there are also negative values (treat as features only)
    total_variance_explained_channels is a list of the total variance explained by the PCA for each channel, if we have an RGB image then we will have 3 values (RGB)
    and for the rest of the channels we will have np.nan
    """
    # Convert PIL image to numpy array if necessary
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Add extra dimension for grayscale images
    if len(image.shape) == 2:
        image = image[..., np.newaxis]

    _, _, c = image.shape

    reconstructed_channels = []
    pca_channels = []
    total_variance_explained_channels = []
    for i in range(c):
        # Get the 2D matrix for the current channel
        channel_data = image[:, :, i]

        channel_data = channel_data.astype(np.float32) / 255.0

        # Apply PCA to the channel data
        pca = PCA(n_components=n_components).fit(channel_data)
        pca_channel = pca.transform(channel_data)
        reconstructed_channel = pca.inverse_transform(pca_channel)

        # Get explained variance ratio for this channel (if it is a color channel)
        if i <= 2:
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            total_variance_explained = cumulative_variance_ratio[-1]
            total_variance_explained_channels.append(total_variance_explained.item())
        else:
            total_variance_explained_channels.append(0)

        reconstructed_channels.append(reconstructed_channel)
        pca_channels.append(pca_channel)

    reconstructed_image = np.stack(reconstructed_channels, axis=-1)
    reconstructed_image = np.clip(reconstructed_image, 0.0, 1.0)

    pca_image = np.stack(pca_channels, axis=-1)

    return reconstructed_image, pca_image, total_variance_explained_channels


def pca_multiple_images(
    images_input: np.ndarray,
    n_components: int = 2,
):
    """
    reconstructed_images are normalized to [0, 1]
    pca_images are not normalized, has the values that occur after applying PCA meaning there are also negative values (treat as features only)
    total_variance_explained_channels is a list of the total variance explained by the PCA for each channel, if we have an RGB image then we will have 3 values (RGB)
    and for the rest of the channels we will have np.nan
    """
    # Reshape images to 2D array (n_samples, h*w, c)
    flattened_images = images_input.reshape(
        images_input.shape[0], -1, images_input.shape[3]
    )

    pca_channels = []
    reconstructed_channels = []
    total_variance_explained_channels = []

    for i in range(images_input.shape[3]):
        channel_data = flattened_images[:, :, i]
        channel_data = channel_data.astype(np.float32) / 255.0
        channel_data = channel_data.T

        # Apply PCA to the channel data
        pca = PCA(n_components=n_components).fit(channel_data)
        pca_channel = pca.transform(channel_data)
        reconstructed_channel = pca.inverse_transform(pca_channel)

        pca_channel = pca_channel.T
        reconstructed_channel = reconstructed_channel.T

        # Get explained variance ratio for this channel (if it is a color channel)
        if i <= 2:
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            total_variance_explained = cumulative_variance_ratio[-1]
            total_variance_explained_channels.append(total_variance_explained.item())
        else:
            total_variance_explained_channels.append(np.nan)

        reconstructed_channels.append(reconstructed_channel)
        pca_channels.append(pca_channel)

    reconstructed_images = np.stack(reconstructed_channels, axis=-1)
    reconstructed_images = np.clip(reconstructed_images, 0.0, 1.0)
    reconstructed_images = reconstructed_images.reshape(images_input.shape)

    pca_images = np.stack(pca_channels, axis=-1)
    pca_images = pca_images.reshape(
        pca_images.shape[0],
        images_input.shape[1],
        images_input.shape[2],
        images_input.shape[3],
    )

    return reconstructed_images, pca_images, total_variance_explained_channels
