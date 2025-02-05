from collections import Counter

from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from PIL import Image

from src.images.utils import load_image, resize_image


# TODO: Speak with Tziola to determine what metrics should be included in the returned dictionary
def analyze_class_distribution(img_json: dict) -> dict:
    """
    Analyzes the distribution of classes in a dataset to detect class imbalances.

    Args:
        img_json: Dictionary containing image paths, path_to_id mapping and label_path

    Returns:
        Updated img_json with class_distribution field added containing distribution metrics
    """
    # Read labels from CSV
    df = pd.read_csv(img_json["label_path"], index_col=False)

    # Filter to only include labels for images in image_paths
    image_names = [
        path.split("/")[-1].split(".")[0] for path in img_json["image_paths"]
    ]
    df = df[df["IMAGE"].isin(image_names)]

    labels = df["LABEL"].tolist()

    class_counts = Counter(labels)
    total_samples = len(labels)

    class_percentages = {
        cls: (count / total_samples) * 100 for cls, count in class_counts.items()
    }

    # Sort classes by frequency
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    counts_array = np.array(list(class_counts.values()))

    stats = {
        "class_counts": dict(class_counts),
        "total_samples": total_samples,
        "num_classes": len(class_counts),
        "mean_samples_per_class": float(np.mean(counts_array)),
        "std_samples_per_class": float(np.std(counts_array)),
        "class_ranking": [cls for cls, _ in sorted_classes],
        "class_percentages": class_percentages,
        "majority_class": [
            cls for cls, count in sorted_classes if count == sorted_classes[0][1]
        ],
        "minority_class": [
            cls for cls, count in sorted_classes if count == sorted_classes[-1][1]
        ],
        # Ratio between the majority and minority class (higher values indicate more imbalance)
        "imbalance_ratio": sorted_classes[0][1] / sorted_classes[-1][1],
        # Shannon entropy of the distribution (lower values indicate more imbalance)
        "class_distribution_entropy": float(
            -np.sum(
                (counts_array / total_samples) * np.log2(counts_array / total_samples)
            )
        ),
    }

    img_json["class_distribution"] = stats

    return img_json


def evaluate_class_imbalance(
    img_json: dict,
    *,
    class_percentage_mild_deviation: float = 0.8,
    class_percentage_severe_deviation: float = 0.5,
) -> dict:
    """
    Evaluates class distribution metrics and provides information on whether action should be taken to balance the classes.
    Args:
        img_json: Dictionary containing image paths, path_to_id mapping and class distribution metrics
        class_percentage_mild_deviation: Float indicating how much the class percentage can deviate from the ideal class percentage before being considered a mild imbalance
        class_percentage_severe_deviation: Float indicating how much the class percentage can deviate from the ideal class percentage before being considered a severe imbalance

    Returns:
        Updated img_json with class_imbalance_evaluation field added containing evaluation results
    """
    if "class_distribution" not in img_json:
        raise ValueError(
            "Class distribution metrics not found in img_json. Run analyze_class_distribution first."
        )

    distribution_stats = img_json["class_distribution"]
    class_specific_imbalances = {}

    num_classes = distribution_stats["num_classes"]

    # Compare against ideal uniform distribution
    ideal_percentage = 100.0 / num_classes
    class_percentages = distribution_stats["class_percentages"]

    # Evaluate each class's deviation from ideal distribution
    for cls, percentage in class_percentages.items():
        ratio_to_ideal = percentage / ideal_percentage
        if ratio_to_ideal < class_percentage_severe_deviation:
            class_specific_imbalances[cls] = "severe_imbalance"
        elif ratio_to_ideal < class_percentage_mild_deviation:
            class_specific_imbalances[cls] = "mild_imbalance"
        else:
            class_specific_imbalances[cls] = "balanced"

    img_json["class_imbalance_evaluation"] = class_specific_imbalances

    return img_json


def oversample_minority_classes(
    img_json: dict,
    *,
    random_state: int = 42,
) -> dict:
    """
    Oversample minority classes using random oversampling and save oversampled images.

    Args:
        img_json: Dictionary containing image paths, path_to_id mapping and labels
        random_state: Random state for reproducible results

    Returns:
        Updated img_json with new oversampled images added
    """
    np.random.seed(random_state)

    df = pd.read_csv(img_json["label_path"], index_col=False)
    original_images = [
        path.split("/")[-1].split(".")[0]
        for path in img_json["image_paths"]
        if "oversampled" not in path
    ]
    df = df[df["IMAGE"].isin(original_images)]

    # Calculate oversampling needed per class
    class_counts = df["LABEL"].value_counts()
    majority_count = class_counts.max()
    classes_to_oversample = {
        label: majority_count - count
        for label, count in class_counts.items()
        if count < majority_count
    }

    # Get original image paths
    original_paths = [
        path
        for path in img_json["image_paths"]
        if "oversampled" not in path.split("/")[-1]
    ]

    # Oversample each minority class
    for class_label, samples_needed in classes_to_oversample.items():
        # Get paths for this class
        class_images = df[df["LABEL"] == class_label]["IMAGE"].tolist()
        class_paths = [
            path
            for path in original_paths
            if path.split("/")[-1].split(".")[0] in class_images
        ]

        # Sample and create new images
        for i, original_path in enumerate(
            np.random.choice(class_paths, size=samples_needed, replace=True)
        ):
            image = load_image(original_path)
            original_name = original_path.split("/")[-1].split(".")[0]
            new_path = f"{original_path.rsplit('.', 1)[0]}_oversampled_{i+1}.{original_path.rsplit('.', 1)[1]}"

            # Save image
            image = Image.fromarray(
                image,
                "RGBA" if len(image.shape) == 3 and image.shape[-1] == 4 else None,
            )
            image.save(new_path)

            # Update metadata
            img_json["image_paths"].append(new_path)
            img_json["path_to_id"][new_path] = (
                f"{img_json['path_to_id'][original_path]}_oversampled"
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "IMAGE": [f"{original_name}_oversampled"],
                            "LABEL": [class_label],
                        }
                    ),
                ],
                ignore_index=True,
            )

    df.to_csv(img_json["label_path"], index=False)

    return img_json


def smote_oversampling(
    img_json: dict, *, random_state: int = 42, k_neighbors: int = 5
) -> dict:
    """
    Applies SMOTE oversampling to balance classes in the dataset.

    Args:
        img_json: Dictionary containing image paths, path_to_id mapping and labels
        random_state: Random state for reproducible results
        k_neighbors: Number of nearest neighbors to use for SMOTE

    Returns:
        Updated img_json with oversampled images added
    """
    # Read labels from CSV
    df = pd.read_csv(img_json["label_path"], index_col=False)

    # Get original image names (excluding previously oversampled)
    original_images = [
        path.split("/")[-1].split(".")[0]
        for path in img_json["image_paths"]
        if "smote" not in path.split("/")[-1]
    ]
    df = df[df["IMAGE"].isin(original_images)]

    # Get original image paths
    original_paths = [
        path for path in img_json["image_paths"] if "smote" not in path.split("/")[-1]
    ]

    # Load and preprocess all images
    images = []
    labels = []
    original_shapes = {}  # Store original shapes for each class
    for img_path in original_paths:
        image = load_image(img_path)
        original_shape = image.shape
        # Convert RGBA to RGB if needed
        if len(image.shape) == 3 and image.shape[-1] == 4:
            image = image[:, :, :3]

        # Store original shape for this class
        img_name = img_path.split("/")[-1].split(".")[0]
        label = df[df["IMAGE"] == img_name]["LABEL"].iloc[0]
        if label not in original_shapes:
            original_shapes[label] = image.shape

        image = resize_image(image)  # Resize to common size
        images.append(image.flatten())
        labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    # Calculate target counts for SMOTE
    class_counts = pd.Series(y).value_counts()
    majority_count = class_counts.max()
    sampling_strategy = {
        label: majority_count
        for label in class_counts.index
        if class_counts[label] < majority_count
    }

    if sampling_strategy:  # Only apply SMOTE if there are minority classes
        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=min(k_neighbors, min(class_counts) - 1),
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Save new SMOTE-generated images
        new_samples_start = len(X)
        for i, (img_array, label) in enumerate(
            zip(X_resampled[new_samples_start:], y_resampled[new_samples_start:])
        ):
            # Get reference original path from same class for naming
            ref_path = [
                p
                for p in original_paths
                if df[df["IMAGE"] == p.split("/")[-1].split(".")[0]]["LABEL"].iloc[0]
                == label
            ][0]
            original_name = ref_path.split("/")[-1].split(".")[0]
            new_path = (
                f"{ref_path.rsplit('.', 1)[0]}_smote_{i+1}.{ref_path.rsplit('.', 1)[1]}"
            )

            # Reshape using the stored original shape for this class
            img_shape = original_shapes[label]
            # First reshape to the size used during SMOTE
            resized_img = resize_image(np.zeros(img_shape, dtype=np.uint8))
            image = img_array.reshape(resized_img.shape)
            # Ensure values are in valid range and convert to uint8
            image = np.clip(image, 0, 255).astype(np.uint8)
            # Convert to PIL Image for resizing back to original shape
            image = Image.fromarray(image)
            image = image.resize((img_shape[1], img_shape[0]), Image.Resampling.LANCZOS)
            image.save(new_path)

            # Update metadata
            img_json["image_paths"].append(new_path)
            img_json["path_to_id"][new_path] = (
                f"{img_json['path_to_id'][ref_path]}_smote"
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {"IMAGE": [f"{original_name}_smote_{i+1}"], "LABEL": [label]}
                    ),
                ],
                ignore_index=True,
            )

    df.to_csv(img_json["label_path"], index=False)

    return img_json
