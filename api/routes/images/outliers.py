import io
from PIL import Image
from typing import List

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from matplotlib import pyplot as plt
import numpy as np

from api.models import (
    ImageOutliers,
    ImageSize,
    ImageOutlierDetectionMethod,
    PixelOutlierDetectionMethod,
    OutlierPixels,
)
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


@router.post(
    "/visualize_image_outliers",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG with the highlighted outlier images.",
        }
    },
    response_class=Response,
)
def visualize_image_outliers_endpoint(
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

        num_images = len(images)
        num_rows = (num_images - 1) // 3 + 1  # Calculate number of rows needed

        plt.figure(figsize=(12, 4 * num_rows))
        plt.suptitle(f"Outlier Images detection using {method}")
        for i, image in enumerate(images):
            img = Image.open(image.file)
            img = np.array(img)

            ax = plt.subplot(num_rows, min(3, num_images), i + 1)
            plt.imshow(img)

            # Add red rectangle around outlier images
            if outlier_labels[i] == -1:  # -1 indicates outlier
                # Create a red rectangle patch with small margin from borders
                margin = 5  # pixels from border
                rect = plt.Rectangle(
                    (0, margin),  # (x,y) of lower left corner
                    img.shape[1] - margin,  # width
                    img.shape[0] - 2 * margin,  # height
                    fill=False,
                    edgecolor="red",
                    linewidth=5,
                )
                ax.add_patch(rect)
                plt.title("Outlier")
            else:
                plt.title("Normal")

            plt.axis("off")
        plt.tight_layout()  # Reduce white margins

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, bbox_inches="tight", format="png")
        img_buffer.seek(0)
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        for image in images:
            image.file.close()

    return Response(content=img_buffer.getvalue(), media_type="image/png")


@router.post(
    "/visualize_pixel_outliers",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG with the highlighted outlier pixels.",
        }
    },
    response_class=Response,
)
def visualize_pixel_outliers_endpoint(
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
        plt.figure(figsize=(10, 8))
        plt.title(f"Outlier pixels detection using {method}")
        plt.imshow(img_array)
        plt.scatter(
            outlier_coordinates[:, 1],
            outlier_coordinates[:, 0],
            color="red",
            marker="o",
            s=20,
        )
        plt.axis("off")
        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, bbox_inches="tight", format="png")
        img_buffer.seek(0)
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        image.file.close()

    return Response(content=img_buffer.getvalue(), media_type="image/png")
