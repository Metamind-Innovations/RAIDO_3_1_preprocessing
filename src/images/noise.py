import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

from src.images.utils import load_image, normalize_pixel_values


def detect_noise(
    img_json: dict,
    *,
    threshold_snr: float = 30,
    threshold_sigma: float = 0.01,
    threshold_local_var: float = 0.01,
) -> dict:
    """
    Analyze whether images need noise removal and add noise analysis results to the json.

    Args:
        img_json: Dictionary containing image paths and path_to_id mapping
        threshold_snr: SNR threshold in dB above which noise removal is unnecessary
        threshold_sigma: Noise standard deviation threshold below which noise removal is unnecessary
        threshold_local_var: Local variance threshold above which noise removal is recommended

    Returns:
        Updated img_json with noise_analysis field added containing analysis results
    """
    noise_analysis = {}

    for img_path in img_json["image_paths"]:
        image = load_image(img_path)

        image = normalize_pixel_values(image)

        results = {}

        # 1. Estimate noise level
        channel_axis = -1 if len(image.shape) > 2 else None
        estimated_sigma = estimate_sigma(image, channel_axis=channel_axis)
        # Handle nested arrays/lists and convert to flat list of floats
        if isinstance(estimated_sigma, (list, np.ndarray)):
            sigma_values = np.array(estimated_sigma).flatten()
        else:
            sigma_values = [estimated_sigma]
        results["estimated_sigma"] = [float(s) for s in sigma_values]

        # 2. Calculate SNR
        mean_signal = np.mean(image)
        noise = image - ndimage.gaussian_filter(image, sigma=1)
        if channel_axis is not None:
            # Calculate SNR for each channel
            noise_std = np.std(noise, axis=(0, 1))
            snr_values = [
                20 * np.log10(mean_signal / std) if std != 0 else float("inf")
                for std in noise_std
            ]
            results["snr_db"] = [float(s) for s in snr_values]
        else:
            # Single channel calculation
            noise_std = np.std(noise)
            snr = (
                20 * np.log10(mean_signal / noise_std)
                if noise_std != 0
                else float("inf")
            )
            results["snr_db"] = [float(snr)]

        # 3. Calculate local variance statistics
        local_var = ndimage.generic_filter(image, np.var, size=5)
        var_stats = {
            "mean_local_var": float(np.mean(local_var)),
            "max_local_var": float(np.max(local_var)),
            "min_local_var": float(np.min(local_var)),
        }
        results["variance_stats"] = var_stats

        # 4. Make recommendation
        # Check if any channel exceeds the thresholds
        needs_denoising = (
            any(sigma > threshold_sigma for sigma in results["estimated_sigma"])
            or any(snr < threshold_snr for snr in results["snr_db"])
            or (var_stats["mean_local_var"] > threshold_local_var)
        )

        results["needs_denoising"] = needs_denoising

        # Store results using image id as key
        img_id = img_json["path_to_id"][img_path]
        noise_analysis[img_id] = results

    img_json["noise_analysis"] = noise_analysis

    return img_json


def denoise_non_local_means(
    img_json: dict,
    *,
    patch_size: int = 7,
    patch_distance: int = 11,
    fast_mode: bool = False,
) -> dict:
    """Apply non-local means denoising to images and store results in img_json.

    Args:
        img_json: Dictionary containing image paths and path_to_id mapping
        patch_size: Size of patches used for denoising
        patch_distance: Maximal distance to search for patches
        fast_mode: If True, use fast version of non-local means

    Returns:
        Updated img_json with denoised images and noise masks added
    """

    for img_path in img_json["image_paths"]:
        image = load_image(img_path)
        image = img_as_float(image)

        # Determine channel_axis based on image dimensions
        channel_axis = -1 if len(image.shape) > 2 else None

        patch_kw = dict(
            patch_size=patch_size,
            patch_distance=patch_distance,
            channel_axis=channel_axis,
        )

        # Apply denoising
        denoised_image = denoise_nl_means(image, fast_mode=fast_mode, **patch_kw)

        # Convert to uint8 and save back to original path
        denoised_uint8 = (denoised_image * 255).astype(np.uint8)
        Image.fromarray(denoised_uint8).save(img_path)

    return img_json


# def visualize_denoised_image(
#     image: Union[np.ndarray, Image.Image],
#     patch_size: int = 7,
#     patch_distance: int = 11,
#     fast_mode: bool = False,
# ) -> None:
#     if not isinstance(image, np.ndarray):
#         image = np.array(image)

#     denoised_image, noise_mask = denoise_non_local_means(
#         image,
#         patch_size,
#         patch_distance,
#         fast_mode,
#     )

#     # Normalize noise mask to [0,1] range
#     noise_mask = (noise_mask - noise_mask.min()) / (noise_mask.max() - noise_mask.min())

#     plt.figure(figsize=(12, 8))
#     plt.suptitle("Non-Local Means Denoising")

#     gs = gridspec.GridSpec(2, 4)
#     gs.update(wspace=0.5)

#     ax1 = plt.subplot(gs[0, :2])
#     ax2 = plt.subplot(gs[0, 2:])
#     ax3 = plt.subplot(gs[1, 1:3])

#     # Plot original image
#     ax1.imshow(image)
#     ax1.set_title("Original Image")
#     ax1.axis("off")

#     # Plot denoised image
#     ax2.imshow(denoised_image)
#     ax2.set_title("Denoised Image")
#     ax2.axis("off")

#     # Plot noise mask
#     ax3.imshow(noise_mask)
#     ax3.set_title("Noise Mask")
#     ax3.axis("off")

#     plt.show()
