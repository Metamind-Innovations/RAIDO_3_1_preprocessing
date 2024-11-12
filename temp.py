import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.images.outliers import detect_pixel_level_outliers
from src.images.dim_reduction import pca_single_image, pca_multiple_images
from src.images.utils import load_and_process_image
from src.images.visualization import plot_multiple_images

from skimage import img_as_float
from skimage.util import random_noise
from src.images.noise import denoise_non_local_means

img_path = "plant_images/outlier_test/fumagina_1.png"

# Load and process the image
img = Image.open(img_path)
img_array = np.array(img)

denoised_img = denoise_non_local_means(img_array, multichannel=True)

plt.figure(figsize=(10, 10))
plt.title("Denoised Image")
plt.imshow(denoised_img)
plt.axis("off")
plt.show()

# TODO: Upload code to gitlab
# NOTE: The API endpoint should maybe return the image instead of the json (?)
