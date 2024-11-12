import io
from PIL import Image
from typing import List

from fastapi import APIRouter, File, UploadFile, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response, JSONResponse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


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


@router.post(
    "/missing_pixels",
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
def plot_missing_pixels_endpoint(
    image: UploadFile = File(description="The image to plot"),
    missing_coords: UploadFile = File(description="The missing coordinates to plot"),
    title: str = Query(None, description="The title of the plot"),
):
    try:
        # Read the uploaded image
        img = Image.open(image.file)
        img_array = np.array(img)

        # Read the uploaded missing coordinates
        coords = np.load(missing_coords.file)

        plt.figure(figsize=(10, 8))
        plt.imshow(img_array)
        plt.scatter(coords[:, 1], coords[:, 0], color="red", marker="o", s=100)
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
        missing_coords.file.close()

    return Response(content=img_buffer.getvalue(), media_type="image/png")


@router.post(
    "/outlier_images",
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
def plot_image_outliers_endpoint(
    images: List[UploadFile] = File(description="The images to plot"),
    outlier_labels: UploadFile = File(description="The outlier labels to plot"),
    title: str = Query("Outlier Images", description="The title of the plot"),
):
    try:
        num_images = len(images)
        num_rows = (num_images - 1) // 3 + 1  # Calculate number of rows needed

        outliers = pd.read_csv(outlier_labels.file, header=None).to_numpy().ravel()

        plt.figure(figsize=(12, 4 * num_rows))
        plt.suptitle(title)
        for i, image in enumerate(images):
            img = Image.open(image.file)
            img = np.array(img)

            ax = plt.subplot(num_rows, min(3, num_images), i + 1)
            plt.imshow(img)

            # Add red rectangle around outlier images
            if outliers[i] == -1:  # -1 indicates outlier
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
        outlier_labels.file.close()

    return Response(content=img_buffer.getvalue(), media_type="image/png")


@router.post(
    "/outlier_pixels",
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
def plot_pixel_outliers_endpoint(
    image: UploadFile = File(description="The image to plot"),
    outlier_coords: UploadFile = File(description="The outlier coordinates to plot"),
    title: str = Query("Outlier Pixels", description="The title of the plot"),
):
    try:
        # Read the uploaded image
        img = Image.open(image.file)
        img_array = np.array(img)

        # Read the uploaded missing coordinates
        coords = np.load(outlier_coords.file)

        plt.figure(figsize=(10, 8))
        plt.imshow(img_array)
        plt.scatter(coords[:, 1], coords[:, 0], color="red", marker="o", s=50)
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
        outlier_coords.file.close()

    return Response(content=img_buffer.getvalue(), media_type="image/png")
