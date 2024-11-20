from collections import Counter

from imblearn.over_sampling import RandomOverSampler
import numpy as np


# TODO: Speak with Tziola to determine what metrics he needs to be included in the returned dictionary
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


# TODO
def oversample_minority_classes(
    images: list[np.ndarray],
    labels: list[str],
) -> tuple[list[np.ndarray], list[str]]:
    pass


# TODO
def smote_oversampling():
    pass
