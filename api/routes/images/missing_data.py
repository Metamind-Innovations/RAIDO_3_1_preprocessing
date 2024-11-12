import io
import json
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
                "application/json": {},
            },
            "description": "Return a JSON file with the imputed image.",
        }
    },
    response_class=Response,
)
def impute_missing_data_endpoint(
    image: UploadFile = File(description="The image to impute missing data"),
    method: ImputationName = Query(..., description="The method to use for imputation"),
):
    try:
        img = Image.open(image.file)
        img_array = np.array(img)

        if method is ImputationName.mean:
            method_name = "mean"
        elif method is ImputationName.median:
            method_name = "median"

        imputed_image = impute_missing_data(img_array, method_name)

        headers = {"Content-Disposition": 'attachment; filename="imputed_image.json"'}

        impute_missing_data_output = {"imputed_image": imputed_image.tolist()}
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        image.file.close()

    return Response(
        json.dumps(impute_missing_data_output),
        headers=headers,
        media_type="application/json",
    )


@router.post(
    "/interpolate_missing_data",
    responses={
        200: {
            "content": {
                "application/json": {},
            },
            "description": "Return a JSON file with the interpolated image.",
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
        img = Image.open(image.file)
        img_array = np.array(img)

        if method is InterpolationName.linear:
            method_name = "linear"
        elif method is InterpolationName.nearest:
            method_name = "nearest"
        elif method is InterpolationName.cubic:
            method_name = "cubic"

        interpolated_image = interpolate_missing_data(img_array, method_name)

        headers = {
            "Content-Disposition": 'attachment; filename="interpolated_image.json"'
        }

        interpolate_missing_data_output = {
            "interpolated_image": interpolated_image.tolist()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        image.file.close()

    return Response(
        json.dumps(interpolate_missing_data_output),
        headers=headers,
        media_type="application/json",
    )


@router.post(
    "/visualize_missing_data",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the missing data highlighted.",
        }
    },
    response_class=Response,
)
def visualize_missing_data_endpoint(
    image: UploadFile = File(description="The image to plot"),
):
    try:
        img = Image.open(image.file)
        img_array = np.array(img)

        missing_coords = detect_missing_data(img_array)

        plt.figure(figsize=(10, 8))
        plt.imshow(img_array)
        plt.title("Missing Data")
        plt.scatter(
            missing_coords[:, 1], missing_coords[:, 0], color="red", marker="o", s=100
        )
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
    "/visualize_imputed_image",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the original and imputed image.",
        }
    },
    response_class=Response,
)
def visualize_imputed_image_endpoint(
    image: UploadFile = File(description="The image to impute missing data"),
    method: ImputationName = Query(..., description="The method to use for imputation"),
):
    try:
        img = Image.open(image.file)
        img_array = np.array(img)

        if method is ImputationName.mean:
            method_name = "mean"
        elif method is ImputationName.median:
            method_name = "median"

        imputed_image = impute_missing_data(img_array, method_name)

        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Imputation using {method_name} method")

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title("Original Image")
        plt.axis("off")

        # Plot imputed image
        plt.subplot(1, 2, 2)
        plt.imshow(imputed_image)
        plt.title("Imputed Image")
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


@router.post(
    "/visualize_interpolated_image",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the original and interpolated image.",
        }
    },
    response_class=Response,
)
def visualize_interpolated_image_endpoint(
    image: UploadFile = File(description="The image to interpolate missing data"),
    method: InterpolationName = Query(
        ..., description="The method to use for interpolation"
    ),
):
    try:
        img = Image.open(image.file)
        img_array = np.array(img)

        if method is InterpolationName.linear:
            method_name = "linear"
        elif method is InterpolationName.nearest:
            method_name = "nearest"
        elif method is InterpolationName.cubic:
            method_name = "cubic"

        interpolated_image = interpolate_missing_data(img_array, method_name)

        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Interpolation using {method_name} method")

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title("Original Image")
        plt.axis("off")

        # Plot interpoalted image
        plt.subplot(1, 2, 2)
        plt.imshow(interpolated_image)
        plt.title("Interpolated Image")
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
