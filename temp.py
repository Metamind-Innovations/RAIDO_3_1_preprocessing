import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.images.outliers import (
    detect_pixel_level_outliers,
    visualize_pixel_outliers,
    visualize_image_outliers,
)
from src.images.dim_reduction import (
    pca_single_image,
    pca_multiple_images,
    visualize_single_reconstructed_image,
    visualize_multiple_reconstructed_images,
    visualize_multiple_pca_images,
)
from src.images.utils import load_and_process_image
from src.images.visualization import plot_multiple_images

from src.images.noise import denoise_non_local_means, visualize_denoised_image
from src.images.missing_data import (
    visualize_imputed_data,
    visualize_interpolated_data,
    visualize_missing_data,
)

from src.images.enrichment import rotate_image, visualize_rotated_images

img_folder = "plant_images/outlier_test"

# Get list of image files in folder
img_files = [f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Load and convert each image to numpy array
images = []
for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)
    img = Image.open(img_path)
    img_array = np.array(img)
    images.append(img_array)

transformed_images = rotate_image(images, negative_angle=-90, positive_angle=90)

visualize_rotated_images(transformed_images, negative_angle=-90, positive_angle=90)

print(len(transformed_images))

for image in transformed_images:
    print(image.shape)
