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

from src.images.enrichment import (
    transform_images,
    visualize_transformed_images,
    normalize_pixel_values,
)

from src.images.balancing import (
    analyze_class_distribution,
    evaluate_class_imbalance,
    oversample_minority_classes,
    smote_oversampling,
)

label_csv = pd.read_csv("plant_images/balance_test/labels.csv")
labels = label_csv["CLASS"].tolist()

# Get list of image files in plant_images directory
image_files = []
for image_id in label_csv["IMAGE_ID"]:
    # Search for files with the image_id as prefix and common image extensions
    for ext in [".jpg", ".jpeg", ".png"]:
        potential_file = os.path.join("plant_images/balance_test", image_id + ext)
        if os.path.exists(potential_file):
            image_files.append(potential_file)
            break

# Load all images into numpy arrays
images = []
for image_file in image_files:
    # Load and convert each image to numpy array
    img = load_and_process_image(image_file)
    images.append(np.array(img))

result = analyze_class_distribution(labels)

class_specific_imbalances = evaluate_class_imbalance(result, 0.1, 0.2)
print(class_specific_imbalances)

# balanced_images, balanced_labels = smote_oversampling(images, labels, k_neighbors=1)

# # Visualize original and balanced dataset
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# # Plot original images
# plot_multiple_images(balanced_images, balanced_labels)


# print("\nClass Distribution Analysis:")
# print("-" * 30)
# print(f"\nTotal Samples: {result['total_samples']}")
# print(f"Number of Classes: {result['num_classes']}")
# print(f"\nMajority Class: {result['majority_class']}")
# print(f"Minority Class: {result['minority_class']}")
# print(f"Imbalance Ratio: {result['imbalance_ratio']:.2f}")

# print("\nClass Counts:")
# for cls, count in result['class_counts'].items():
#     print(f"{cls:25} {count:5d} ({result['class_percentages'][cls]:.1f}%)")

# print(f"\nMean Samples per Class: {result['mean_samples_per_class']:.1f}")
# print(f"Std Dev Samples per Class: {result['std_samples_per_class']:.1f}")
# print(f"Distribution Entropy: {result['class_distribution_entropy']:.2f}")
