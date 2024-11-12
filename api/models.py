from enum import Enum
from typing import List, Tuple, Any

import numpy as np
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
