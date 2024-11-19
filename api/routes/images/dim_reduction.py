import io
import json
from PIL import Image
from typing import List

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from matplotlib import pyplot as plt
import numpy as np

from api.models import ImageSize
from src.images.dim_reduction import pca_single_image, pca_multiple_images
from src.images.utils import resize_image


router = APIRouter(
    prefix="/images/dim_reduction", tags=["Images Dimensionality Reduction"]
)


@router.post(
    "/single_image_pca",
    responses={
        200: {
            "content": {
                "application/json": {},
            },
            "description": "Return a JSON file with the PCA-related results.",
        }
    },
    response_class=Response,
)
def single_image_pca_endpoint(
    image: UploadFile = File(description="The image to perform PCA on"),
    n_components: int = Query(30, description="The number of components to keep"),
):
    try:
        img = Image.open(image.file)
        img_array = np.array(img)

        reconstructed_image, pca_image, total_variance_explained_channels = (
            pca_single_image(img_array, n_components)
        )
        single_image_pca_output = {
            "reconstructed_image": reconstructed_image.tolist(),
            "pca_image": pca_image.tolist(),
            "total_variance_explained_channels": [
                float(x) for x in total_variance_explained_channels
            ],
        }
        headers = {
            "Content-Disposition": 'attachment; filename="single_image_pca.json"'
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        image.file.close()

    return Response(
        json.dumps(single_image_pca_output),
        headers=headers,
        media_type="application/json",
    )


@router.post(
    "/multiple_images_pca",
    responses={
        200: {
            "content": {
                "application/json": {},
            },
            "description": "Return a JSON file with the PCA-related results.",
        }
    },
    response_class=Response,
)
def multiple_images_pca_endpoint(
    images: List[UploadFile] = File(description="The images to perform PCA on"),
    n_components: int = Query(3, description="The number of components to keep"),
    image_size: ImageSize = Query(..., description="The size of the resized images"),
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

        images_array = np.stack(processed_images)

        reconstructed_images, pca_images, total_variance_explained_channels = (
            pca_multiple_images(images_array, n_components)
        )

        multiple_images_pca_output = {
            "reconstructed_images": reconstructed_images.tolist(),
            "pca_images": pca_images.tolist(),
            "total_variance_explained_channels": [
                float(x) for x in total_variance_explained_channels
            ],
        }
        headers = {
            "Content-Disposition": 'attachment; filename="multiple_images_pca.json"'
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        for image in images:
            image.file.close()

    return Response(
        json.dumps(multiple_images_pca_output),
        headers=headers,
        media_type="application/json",
    )


@router.post(
    "/visualize_single_reconstructed_image",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG with the reconstructed image.",
        }
    },
    response_class=Response,
)
def visualize_single_reconstructed_image_endpoint(
    image: UploadFile = File(description="The image to perform PCA on"),
    n_components: int = Query(30, description="The number of components to keep"),
):
    try:
        img = Image.open(image.file)
        img_array = np.array(img)

        reconstructed_image, _, _ = pca_single_image(img_array, n_components)

        plt.figure(figsize=(12, 6))
        plt.suptitle("Single-Image reconstruction using PCA components")

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title("Original Image")
        plt.axis("off")

        # Plot denoised image
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image)
        plt.title("Reconstructed Image")
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
    "/visualize_multiple_reconstructed_images",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG with the reconstructed images.",
        }
    },
    response_class=Response,
)
def visualize_multiple_reconstructed_images_endpoint(
    images: List[UploadFile] = File(description="The images to perform PCA on"),
    n_components: int = Query(3, description="The number of components to keep"),
    image_size: ImageSize = Query(..., description="The size of the resized images"),
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

        images_array = np.stack(processed_images)

        reconstructed_images, _, _ = pca_multiple_images(images_array, n_components)

        num_images = len(images)
        num_rows = (
            num_images - 1
        ) // 3 + 1  # Calculate rows needed for original images
        total_rows = num_rows * 2  # Double the rows to accommodate reconstructed images

        plt.figure(figsize=(12, 4 * total_rows))
        plt.suptitle("Original vs Reconstructed Images using PCA")

        # Plot original images
        for i in range(num_images):
            plt.subplot(total_rows, 3, i + 1)
            plt.imshow(images_array[i])
            plt.title(f"Original {i+1}")
            plt.axis("off")

        # Plot reconstructed images starting from new row
        start_idx = num_rows * 3  # Start index for reconstructed images
        for i in range(num_images):
            plt.subplot(total_rows, 3, start_idx + i + 1)
            plt.imshow(reconstructed_images[i])
            plt.title(f"Reconstructed {i+1}")
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
    "/visualize_multiple_pca_images",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG with the eigen images.",
        }
    },
    response_class=Response,
)
def visualize_multiple_pca_images_endpoint(
    images: List[UploadFile] = File(description="The images to perform PCA on"),
    n_components: int = Query(3, description="The number of components to keep"),
    image_size: ImageSize = Query(..., description="The size of the resized images"),
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

        images_array = np.stack(processed_images)

        _, pca_images, _ = pca_multiple_images(images_array, n_components)

        num_images = len(images)
        num_rows = (num_images - 1) // 3 + 1  # Calculate number of rows needed

        plt.figure(figsize=(12, 4 * num_rows))
        plt.suptitle("Eigen Images using PCA")
        for i, pca_image in enumerate(pca_images):
            plt.subplot(num_rows, min(3, num_images), i + 1)
            normalized_image = (pca_image - pca_image.min()) / (
                pca_image.max() - pca_image.min()
            )
            plt.imshow(normalized_image)
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
