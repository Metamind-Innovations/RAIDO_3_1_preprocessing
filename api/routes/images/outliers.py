from PIL import Image
from typing import List

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import numpy as np

from api.models import ImageOutliers, ImageSize, ImageOutlierDetectionMethod, PixelOutlierDetectionMethod, OutlierPixels
from src.images.outliers import detect_image_level_outliers, detect_pixel_level_outliers
from src.images.utils import resize_image


router = APIRouter(prefix="/images/outliers", tags=["Images Outliers"])


@router.post(
    "/detect_image_outliers",
    response_model=ImageOutliers,
)
def detect_image_level_outliers_endpoint(
    images: List[UploadFile] = File(description="The images to detect outliers in"),
    image_size: ImageSize = Query(..., description="The size of the resized images"),
    method: ImageOutlierDetectionMethod = Query(
        ImageOutlierDetectionMethod.isolation_forest,
        description="The method to use for image outlier detection",
    ),
    n_estimators: int = Query(
        100,
        description="The number of trees in the forest for Isolation Forest",
    ),
    contamination: float = Query(
        0.1,
        description="The contamination parameter for Isolation Forest (between 0 and 1)",
        ge=0,
        le=1,
    ),
    random_state: int = Query(
        42,
        description="The random state for Isolation Forest",
    ),
):
    try:
        processed_images = []
        for image in images:
            img = Image.open(image.file)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img)
            img_array = resize_image(img_array, height=image_size, width=image_size)
            processed_images.append(img_array)

        # Stack into single array
        images_array = np.stack(processed_images)
        
        if method is ImageOutlierDetectionMethod.isolation_forest:
            method_name = "isolation_forest"

        outlier_labels = detect_image_level_outliers(
            images_array,
            method_name,
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        for image in images:
            image.file.close()

    return ImageOutliers(outlier_labels=outlier_labels.tolist())


@router.post(
    "/detect_pixel_outliers",
    response_model=OutlierPixels,
)
def detect_pixel_level_outliers_endpoint(
    image: UploadFile = File(description="The image to detect outliers in"),
    method: PixelOutlierDetectionMethod = Query(
        PixelOutlierDetectionMethod.lof,
        description="The method to use for pixel outlier detection",
    ),
    n_neighbors: int = Query(
        20,
        description="The number of neighbors to use for Local Outlier Factor",
    ),
    contamination: float = Query(
        0.1,
        description="The contamination parameter for Local Outlier Factor (between 0 and 1)",
        ge=0,
        le=1,
    ),
):
    try:
        img = Image.open(image.file)
        img_array = np.array(img)

        if method is PixelOutlierDetectionMethod.lof:
            method_name = "lof"

        outlier_coordinates = detect_pixel_level_outliers(
            img_array, method_name, n_neighbors=n_neighbors, contamination=contamination
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        image.file.close()

    return OutlierPixels(outlier_coordinates=outlier_coordinates.tolist())
