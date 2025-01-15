from enum import Enum
from typing import Dict, List, Tuple

from pydantic import BaseModel


class MissingPixels(BaseModel):
    missing_coordinates: List[Tuple[int, int]] = []

    model_config = {
        "json_schema_extra": {
            "examples": [{"missing_coordinates": [(10, 20), (30, 40)]}]
        }
    }


class ImputationName(str, Enum):
    mean = "mean"
    median = "median"


class InterpolationName(str, Enum):
    linear = "linear"
    nearest = "nearest"
    cubic = "cubic"


class ImageOutliers(BaseModel):
    outlier_labels: List[int] = []

    model_config = {
        "json_schema_extra": {"examples": [{"outlier_labels": [-1, 1, -1, 1]}]}
    }


class ImageSize(int, Enum):
    small = 360
    medium = 480
    large = 640


class ImageOutlierDetectionMethod(str, Enum):
    isolation_forest = "Isolation Forest"


class PixelOutlierDetectionMethod(str, Enum):
    lof = "Local Outlier Factor"


class OutlierPixels(BaseModel):
    outlier_coordinates: List[Tuple[int, int]] = []

    model_config = {
        "json_schema_extra": {
            "examples": [{"outlier_coordinates": [(10, 20), (30, 40)]}]
        }
    }


class DenoiseMode(str, Enum):
    fast = "fast"
    slow = "slow"


class ImageTransformation(str, Enum):
    Rotation = "rotation"
    crop = "crop"
    resize = "resize"
    shear = "shear"
    brightness = "brightness"
    contrast = "contrast"
    saturation = "saturation"
    hue = "hue"
    vertical_flip = "vertical_flip"
    horizontal_flip = "horizontal_flip"


class ClassDistributionAnalysis(BaseModel):
    class_counts: Dict[str, int] = {}
    total_samples: int = 0
    num_classes: int = 0
    mean_samples_per_class: float = 0.0
    std_samples_per_class: float = 0.0
    class_ranking: List[str] = []
    class_percentages: Dict[str, float] = {}
    majority_class: List[str] = []
    minority_class: List[str] = []
    imbalance_ratio: float = 0.0
    class_distribution_entropy: float = 0.0

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "class_counts": {"class1": 10, "class2": 20, "class3": 30},
                    "total_samples": 60,
                    "num_classes": 3,
                    "mean_samples_per_class": 20.0,
                    "std_samples_per_class": 10.0,
                    "class_ranking": ["class1", "class2", "class3"],
                    "class_percentages": {
                        "class1": 0.16666666666666666,
                        "class2": 0.3333333333333333,
                        "class3": 0.5,
                    },
                    "majority_class": ["class1"],
                    "minority_class": ["class3"],
                    "imbalance_ratio": 0.5,
                    "class_distribution_entropy": 1.0986122886681098,
                }
            ]
        }
    }


class ClassImbalanceEvaluation(BaseModel):
    class_specific_imbalances: Dict[str, str] = {}

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "class_specific_imbalances": {
                        "class1": "severe_imbalance",
                        "class2": "mild_imbalance",
                        "class3": "balanced",
                    }
                }
            ]
        }
    }
