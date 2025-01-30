import numpy as np
from sklearn.ensemble import IsolationForest

from src.images.utils import load_image, resize_image


def detect_image_level_outliers(
    img_json: dict,
    method: str = "isolation_forest",
    *,
    n_estimators: int = 100,
    contamination: float = 0.1,
    random_state: int = 42,
    height: int = 360,
    width: int = 360,
) -> dict:
    """
    Detect outlier images using unsupervised outlier detection algorithms.

    Args:
        img_json: Dictionary containing image paths and path_to_id mapping
        method: Outlier detection method. Currently supported methods:
            - "isolation_forest": Uses Isolation Forest algorithm
        n_estimators: Number of base estimators in the ensemble for Isolation Forest
        contamination: Expected proportion of outliers in the dataset, between 0 and 0.5
        random_state: Random state for reproducible results

    Returns:
        Updated img_json with image_outliers field added containing paths to outlier images
    """
    if method not in ["isolation_forest"]:
        raise NotImplementedError(f"Invalid method: {method}")

    # Load all images and stack them
    images = []
    for img_path in img_json["image_paths"]:
        image = load_image(img_path)
        image = resize_image(image, height, width)
        # Convert RGBA to RGB if necessary
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        print(image.shape)
        images.append(image)
    
    images_array = np.stack(images)
    
    # Reshape images to 2D array (n_samples, n_features)
    n_samples = images_array.shape[0]
    flattened_images = images_array.reshape(n_samples, -1)

    if method == "isolation_forest":
        iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        outlier_labels = iso_forest.fit_predict(flattened_images)

    # Get paths of outlier images
    outlier_paths = [
        path for path, label in zip(img_json["image_paths"], outlier_labels)
        if label == -1
    ]

    img_json["image_outliers"] = outlier_paths

    return img_json


def remove_image_outliers(img_json: dict) -> dict:
    """
    Remove outlier images from the img_json
    """
    img_json["image_paths"] = [
        path for path in img_json["image_paths"] if path not in img_json["image_outliers"]
    ]
    return img_json


# def visualize_image_outliers(
#     images_input: np.ndarray,
#     method: str = "isolation_forest",
#     *,
#     n_estimators: int = 100,
#     contamination: float = 0.1,
#     random_state: int = 42,
# ):
#     outlier_labels = detect_image_level_outliers(
#         images_input,
#         method,
#         n_estimators=n_estimators,
#         contamination=contamination,
#         random_state=random_state,
#     )

#     num_images = len(images_input)
#     num_rows = (num_images - 1) // 3 + 1  # Calculate number of rows needed

#     plt.figure(figsize=(12, 4 * num_rows))
#     plt.suptitle(f"Outlier Images detection using {method}")
#     for i, image in enumerate(images_input):
#         ax = plt.subplot(num_rows, min(3, num_images), i + 1)
#         plt.imshow(image)

#         # Add red rectangle around outlier images
#         if outlier_labels[i] == -1:  # -1 indicates outlier
#             # Create a red rectangle patch with small margin from borders
#             margin = 5  # pixels from border
#             rect = plt.Rectangle(
#                 (0, margin),  # (x,y) of lower left corner
#                 image.shape[1] - margin,  # width
#                 image.shape[0] - 2 * margin,  # height
#                 fill=False,
#                 edgecolor="red",
#                 linewidth=5,
#             )
#             ax.add_patch(rect)
#             plt.title("Outlier")
#         else:
#             plt.title("Normal")

#         plt.axis("off")
#     plt.tight_layout()
#     plt.show()


# def visualize_pixel_outliers(
#     image: Union[np.ndarray, Image.Image],
#     method: str = "lof",
#     *,
#     n_neighbors: int = 20,
#     contamination: float = 0.1,
# ):
#     if not isinstance(image, np.ndarray):
#         image = np.array(image)

#     outlier_coords = detect_pixel_level_outliers(
#         image, method, n_neighbors=n_neighbors, contamination=contamination
#     )

#     plt.figure(figsize=(10, 8))
#     plt.title(f"Outlier pixels detection using {method}")
#     plt.imshow(image)
#     plt.scatter(
#         outlier_coords[:, 1],
#         outlier_coords[:, 0],
#         color="red",
#         marker="o",
#         s=20,
#     )
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()
