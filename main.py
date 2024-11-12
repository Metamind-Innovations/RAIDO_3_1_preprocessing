import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from src.images.missing_data import (
    detect_missing_data,
    impute_missing_data,
    interpolate_missing_data,
)
from src.images.visualization import plot_image_outliers, plot_multiple_images
from src.images.utils import load_and_process_image
from src.images.outliers import detect_image_level_outliers


def main():
    image_folder = "plant_images/outlier_test"

    # Get list of image files in the folder
    image_files = [
        f
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Load and process each image
    processed_images = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        processed_image = load_and_process_image(image_path)
        processed_images.append(processed_image)

    # Convert list to numpy array
    processed_images = np.array(processed_images)

    # plot_multiple_images(processed_images)

    outlier_labels = detect_image_level_outliers(
        processed_images,
        method="isolation_forest",
        contamination=0.1,
        random_state=42,
        n_estimators=20,
    )

    print(outlier_labels)

    plot_image_outliers(processed_images, outlier_labels)

    # # Get indices of outlier images (where labels are -1)
    # outlier_indices = np.where(outlier_labels == -1)[0]

    # # Extract outlier images
    # outlier_images = processed_images[outlier_indices]

    # # Display each outlier image
    # for i, img in enumerate(outlier_images):
    #     print(f"Outlier image {i+1} (index {outlier_indices[i]})")
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(img)
    #     plt.title(f"Outlier image {i+1} (index {outlier_indices[i]})")
    #     plt.axis("off")
    #     plt.show()


if __name__ == "__main__":
    main()
