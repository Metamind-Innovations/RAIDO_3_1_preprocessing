import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.images.outliers import detect_pixel_level_outliers
from src.images.dim_reduction import pca_single_image, pca_multiple_images
from src.images.utils import load_and_process_image
from src.images.visualization import plot_multiple_images

from src.images.noise import denoise_non_local_means, visualize_denoised_image
from src.images.missing_data import visualize_imputed_data, visualize_interpolated_data, visualize_missing_data

img_path = "plant_images/fumagina_1_noisy.png"

image = Image.open(img_path)
image_array = np.array(image)

visualize_denoised_image(image_array, fast_mode=True, multichannel=True)