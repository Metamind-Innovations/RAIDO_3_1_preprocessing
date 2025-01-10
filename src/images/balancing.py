from collections import Counter

from imblearn.over_sampling import RandomOverSampler, SMOTE
import numpy as np


# TODO: Speak with Tziola to determine what metrics should be included in the returned dictionary
def analyze_class_distribution(labels: list[str]) -> dict:
    """
    Analyzes the distribution of classes in a dataset to detect class imbalances.

    Args:
        labels: List of class labels

    Returns:
        dict: Dictionary containing various class distribution metrics
    """

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

    return stats


def evaluate_class_imbalance(
    distribution_stats: dict,
    class_percentage_mild_deviation: float,
    class_percentage_severe_deviation: float,
) -> dict:
    """
    Evaluates class distribution metrics and provides information on whether action should be taken to balance the classes.
    Args:
        distribution_stats: Dictionary containing class distribution metrics (from analyze_class_distribution)
        class_percentage_mild_deviation: Float indicating how much the class percentage can deviate from the ideal class percentage before being considered a mild imbalance
        class_percentage_severe_deviation: Float indicating how much the class percentage can deviate from the ideal class percentage before being considered a severe imbalance

    Returns:
        dict: Evaluation results on imbalance detection
    """
    class_specific_imbalances = {}

    num_classes = distribution_stats["num_classes"]

    # Compare against ideal uniform distribution
    ideal_percentage = 100.0 / num_classes
    class_percentages = distribution_stats["class_percentages"]

    print(ideal_percentage)
    print(class_percentages)

    # Evaluate each class's deviation from ideal distribution
    for cls, percentage in class_percentages.items():
        ratio_to_ideal = percentage / ideal_percentage
        if ratio_to_ideal < class_percentage_severe_deviation:
            class_specific_imbalances[cls] = "severe_imbalance"
        elif ratio_to_ideal < class_percentage_mild_deviation:
            class_specific_imbalances[cls] = "mild_imbalance"
        else:
            class_specific_imbalances[cls] = "balanced"

    return class_specific_imbalances


def oversample_minority_classes(
    images: list[np.ndarray],
    labels: list[str],
) -> tuple[list[np.ndarray], list[str]]:
    X = np.array(images)
    # Flatten images
    if len(X.shape) > 2:
        original_shape = X.shape[1:]
        X = X.reshape(X.shape[0], -1)
    else:
        original_shape = None

    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, labels)

    # Reshape images back to original shape
    if original_shape is not None:
        X_resampled = X_resampled.reshape(-1, *original_shape)

    return list(X_resampled), list(y_resampled)


def smote_oversampling(
    images: list[np.ndarray],
    labels: list[str],
    k_neighbors: int = 5,
) -> tuple[list[np.ndarray], list[str]]:
    X = np.array(images)
    # Flatten images
    if len(X.shape) > 2:
        original_shape = X.shape[1:]
        X = X.reshape(X.shape[0], -1)
    else:
        original_shape = None

    # Check if we have enough samples in each class for SMOTE
    class_counts = Counter(labels)
    min_samples = min(class_counts.values())

    if min_samples < 2:
        raise ValueError(
            f"SMOTE requires at least 2 samples per class. "
            f"Minimum samples found: {min_samples}"
        )

    if k_neighbors >= min_samples:
        raise ValueError(
            f"k_neighbors ({k_neighbors}) must be less than the number of "
            f"samples in the smallest class ({min_samples})"
        )

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, labels)

    # Reshape images back to original shape
    if original_shape is not None:
        X_resampled = X_resampled.reshape(-1, *original_shape)

    return list(X_resampled), list(y_resampled)
