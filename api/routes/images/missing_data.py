import io
from PIL import Image

from fastapi import APIRouter, File, UploadFile, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response, JSONResponse
from matplotlib import pyplot as plt
import numpy as np


from src.images.missing_data import (
    detect_missing_data,
    impute_missing_data,
    interpolate_missing_data,
)
from api.models import MissingPixels, ImputationName, InterpolationName


router = APIRouter(prefix="/images/missing_data", tags=["Images Missing Data"])


@router.post(
    "/detect_missing_pixels",
    response_model=MissingPixels,
)
def detect_missing_data_endpoint(
    image: UploadFile = File(description="The image to check for missing data"),
):
    """Detect missing data in an uploaded image"""
    try:
        # Read the uploaded image
        img = Image.open(image.file)
        img_array = np.array(img)

        missing_coords = detect_missing_data(img_array)
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        image.file.close()

    return MissingPixels(missing_coordinates=missing_coords.tolist())


@router.post(
    "/impute_missing_data",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the imputed image.",
        }
    },
    response_class=Response,
)
def impute_missing_data_endpoint(
    image: UploadFile = File(description="The image to impute missing data"),
    method: ImputationName = Query(..., description="The method to use for imputation"),
):
    """Impute missing data in an uploaded image"""
    try:
        # Read the uploaded image
        img = Image.open(image.file)
        img_array = np.array(img)

        if method is ImputationName.mean:
            method_name = "mean"
        elif method is ImputationName.median:
            method_name = "median"

        missing_coords = detect_missing_data(img_array)
        imputed_image = impute_missing_data(img_array, missing_coords, method_name)

        plt.figure(figsize=(10, 8))
        plt.imshow(imputed_image)
        plt.title(f"Imputed image using {method_name} method")
        plt.axis("off")

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


@router.post(
    "/interpolate_missing_data",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the imputed image.",
        }
    },
    response_class=Response,
)
def interpolate_missing_data_endpoint(
    image: UploadFile = File(description="The image to interpolate missing data"),
    method: InterpolationName = Query(
        ..., description="The method to use for interpolation"
    ),
):
    try:
        # Read the uploaded image
        img = Image.open(image.file)
        img_array = np.array(img)

        if method is InterpolationName.linear:
            method_name = "linear"
        elif method is InterpolationName.nearest:
            method_name = "nearest"
        elif method is InterpolationName.cubic:
            method_name = "cubic"

        missing_coords = detect_missing_data(img_array)
        interpolated_image = interpolate_missing_data(
            img_array, missing_coords, method_name
        )

        plt.figure(figsize=(10, 8))
        plt.imshow(interpolated_image)
        plt.title(f"Interpolated image using {method_name} method")
        plt.axis("off")

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
