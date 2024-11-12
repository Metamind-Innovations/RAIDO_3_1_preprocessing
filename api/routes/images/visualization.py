import io
from PIL import Image
from typing import List

from fastapi import APIRouter, File, UploadFile, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response, JSONResponse
from matplotlib import pyplot as plt
import numpy as np


router = APIRouter(prefix="/images/visualization", tags=["Images Visualization"])


@router.post(
    "/image",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the plots.",
        }
    },
    response_class=Response,
)
def plot_image_endpoint(
    image: UploadFile = File(description="The image to plot"),
    title: str = Query(None, description="The title of the plot"),
):
    try:
        # Read the uploaded image
        img = Image.open(image.file)
        img_array = np.array(img)

        plt.figure(figsize=(10, 8))
        plt.imshow(img_array)
        if title:
            plt.title(title)
        plt.axis("off")

        # Save the figure to a PNG file
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
    "/multiple_images",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the plots.",
        }
    },
    response_class=Response,
)
def plot_multiple_images_endpoint(
    images: List[UploadFile] = File(description="The images to plot"),
    title: str = Query(None, description="The title of the plot"),
):
    try:
        num_images = len(images)
        num_rows = (num_images - 1) // 3 + 1  # Calculate number of rows needed

        plt.figure(figsize=(12, 4 * num_rows))
        plt.suptitle(title)
        for i, image in enumerate(images):
            img = Image.open(image.file)
            img_array = np.array(img)

            plt.subplot(num_rows, min(3, num_images), i + 1)
            plt.imshow(img_array)
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
        for image in images:
            image.file.close()

    return Response(content=img_buffer.getvalue(), media_type="image/png")
