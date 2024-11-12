import json
from PIL import Image
from typing import List

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
import numpy as np

from api.models import ImageSize
from src.images.dim_reduction import pca_single_image, pca_multiple_images
from src.images.utils import resize_image


router = APIRouter(prefix="/images/dim_reduction", tags=["Images Dimensionality Reduction"])


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

        reconstructed_image, pca_image, total_variance_explained_channels = pca_single_image(img_array, n_components)
        single_image_pca_output = {
            "reconstructed_image": reconstructed_image.tolist(),
            "pca_image": pca_image.tolist(),
            "total_variance_explained_channels": [float(x) for x in total_variance_explained_channels]
        }
        headers = {'Content-Disposition': 'attachment; filename="single_image_pca.json"'}
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        image.file.close()

    return Response(json.dumps(single_image_pca_output), headers=headers, media_type="application/json")

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
    n_components: int = Query(30, description="The number of components to keep"),
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

        reconstructed_images, pca_images, total_variance_explained_channels = pca_multiple_images(images_array, n_components)

        multiple_images_pca_output = {
            "reconstructed_images": reconstructed_images.tolist(),
            "pca_images": pca_images.tolist(),
            "total_variance_explained_channels": [float(x) for x in total_variance_explained_channels]
        }
        headers = {'Content-Disposition': 'attachment; filename="multiple_images_pca.json"'}
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        for image in images:
            image.file.close()

    return Response(json.dumps(multiple_images_pca_output), headers=headers, media_type="application/json")
