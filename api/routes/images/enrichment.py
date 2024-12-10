import io
import json
from PIL import Image
from typing import List, Optional

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from matplotlib import pyplot as plt
import numpy as np

from api.models import ImageTransformation
from src.images.enrichment import transform_images, normalize_pixel_values


router = APIRouter(prefix="/images/enrichment", tags=["Images Enrichment"])


@router.post(
    "/transform_images",
    responses={
        200: {
            "content": {
                "application/json": {},
            },
            "description": "Return a JSON file with the transformed images.",
        }
    },
    response_class=Response,
)
def transform_images_endpoint(
    images: List[UploadFile] = File(description="The images to transform"),
    transformations: List[ImageTransformation] = Query(
        ..., description="The transformations to perform"
    ),
    # Rotation parameters
    negative_angle: Optional[int] = Query(
        None, description="Minimum rotation angle (default: -90)"
    ),
    positive_angle: Optional[int] = Query(
        None, description="Maximum rotation angle (default: 90)"
    ),
    # Crop parameters
    min_crop_height: Optional[int] = Query(
        None, description="Minimum crop height (default: 180)"
    ),
    max_crop_height: Optional[int] = Query(
        None, description="Maximum crop height (default: 320)"
    ),
    w2h_ratio: Optional[float] = Query(
        None, description="Width to height ratio for crop (default: 1.0)"
    ),
    # Resize parameters
    height: Optional[int] = Query(
        None, description="Target height for resize (default: 320)"
    ),
    width: Optional[int] = Query(
        None, description="Target width for resize (default: 180)"
    ),
    # Shear parameters
    min_shear_angle: Optional[int] = Query(
        None, description="Minimum shear angle (default: -45)"
    ),
    max_shear_angle: Optional[int] = Query(
        None, description="Maximum shear angle (default: 45)"
    ),
    # Color adjustment parameters
    brightness_min_factor: Optional[float] = Query(
        None, description="Minimum brightness factor (default: 0.2)"
    ),
    brightness_max_factor: Optional[float] = Query(
        None, description="Maximum brightness factor (default: 2.0)"
    ),
    contrast_min_factor: Optional[float] = Query(
        None, description="Minimum contrast factor (default: 0.2)"
    ),
    contrast_max_factor: Optional[float] = Query(
        None, description="Maximum contrast factor (default: 2.0)"
    ),
    saturation_min_factor: Optional[float] = Query(
        None, description="Minimum saturation factor (default: 0.2)"
    ),
    saturation_max_factor: Optional[float] = Query(
        None, description="Maximum saturation factor (default: 2.0)"
    ),
    hue_min_factor: Optional[float] = Query(
        None, description="Minimum hue factor (default: -0.5)"
    ),
    hue_max_factor: Optional[float] = Query(
        None, description="Maximum hue factor (default: 0.5)"
    ),
):
    try:
        images_array = []
        for image in images:
            img = Image.open(image.file)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img)
            images_array.append(img_array)

        transform_kwargs = {
            k: v
            for k, v in {
                "negative_angle": negative_angle,
                "positive_angle": positive_angle,
                "min_crop_height": min_crop_height,
                "max_crop_height": max_crop_height,
                "w2h_ratio": w2h_ratio,
                "height": height,
                "width": width,
                "min_shear_angle": min_shear_angle,
                "max_shear_angle": max_shear_angle,
                "brightness_min_factor": brightness_min_factor,
                "brightness_max_factor": brightness_max_factor,
                "contrast_min_factor": contrast_min_factor,
                "contrast_max_factor": contrast_max_factor,
                "saturation_min_factor": saturation_min_factor,
                "saturation_max_factor": saturation_max_factor,
                "hue_min_factor": hue_min_factor,
                "hue_max_factor": hue_max_factor,
            }.items()
            if v is not None
        }

        transformed_images = transform_images(
            images_array, transformations, **transform_kwargs
        )
        transformed_images = [img.tolist() for img in transformed_images]

        transformed_images_output = {
            "transformed_images": transformed_images,
        }
        headers = {
            "Content-Disposition": 'attachment; filename="transformed_images.json"'
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        for image in images:
            image.file.close()

    return Response(
        json.dumps(transformed_images_output),
        headers=headers,
        media_type="application/json",
    )


@router.post(
    "/visualize_transformed_images",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG with the transformed images.",
        }
    },
    response_class=Response,
)
def visualize_transformed_images_endpoint(
    images: List[UploadFile] = File(description="The images to transform"),
    transformations: List[ImageTransformation] = Query(
        ..., description="The transformations to perform"
    ),
    # Rotation parameters
    negative_angle: Optional[int] = Query(
        None, description="Minimum rotation angle (default: -90)"
    ),
    positive_angle: Optional[int] = Query(
        None, description="Maximum rotation angle (default: 90)"
    ),
    # Crop parameters
    min_crop_height: Optional[int] = Query(
        None, description="Minimum crop height (default: 180)"
    ),
    max_crop_height: Optional[int] = Query(
        None, description="Maximum crop height (default: 320)"
    ),
    w2h_ratio: Optional[float] = Query(
        None, description="Width to height ratio for crop (default: 1.0)"
    ),
    # Resize parameters
    height: Optional[int] = Query(
        None, description="Target height for resize (default: 320)"
    ),
    width: Optional[int] = Query(
        None, description="Target width for resize (default: 180)"
    ),
    # Shear parameters
    min_shear_angle: Optional[int] = Query(
        None, description="Minimum shear angle (default: -45)"
    ),
    max_shear_angle: Optional[int] = Query(
        None, description="Maximum shear angle (default: 45)"
    ),
    # Color adjustment parameters
    brightness_min_factor: Optional[float] = Query(
        None, description="Minimum brightness factor (default: 0.2)"
    ),
    brightness_max_factor: Optional[float] = Query(
        None, description="Maximum brightness factor (default: 2.0)"
    ),
    contrast_min_factor: Optional[float] = Query(
        None, description="Minimum contrast factor (default: 0.2)"
    ),
    contrast_max_factor: Optional[float] = Query(
        None, description="Maximum contrast factor (default: 2.0)"
    ),
    saturation_min_factor: Optional[float] = Query(
        None, description="Minimum saturation factor (default: 0.2)"
    ),
    saturation_max_factor: Optional[float] = Query(
        None, description="Maximum saturation factor (default: 2.0)"
    ),
    hue_min_factor: Optional[float] = Query(
        None, description="Minimum hue factor (default: -0.5)"
    ),
    hue_max_factor: Optional[float] = Query(
        None, description="Maximum hue factor (default: 0.5)"
    ),
):
    try:
        images_array = []
        for image in images:
            img = Image.open(image.file)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img)
            images_array.append(img_array)

        transform_kwargs = {
            k: v
            for k, v in {
                "negative_angle": negative_angle,
                "positive_angle": positive_angle,
                "min_crop_height": min_crop_height,
                "max_crop_height": max_crop_height,
                "w2h_ratio": w2h_ratio,
                "height": height,
                "width": width,
                "min_shear_angle": min_shear_angle,
                "max_shear_angle": max_shear_angle,
                "brightness_min_factor": brightness_min_factor,
                "brightness_max_factor": brightness_max_factor,
                "contrast_min_factor": contrast_min_factor,
                "contrast_max_factor": contrast_max_factor,
                "saturation_min_factor": saturation_min_factor,
                "saturation_max_factor": saturation_max_factor,
                "hue_min_factor": hue_min_factor,
                "hue_max_factor": hue_max_factor,
            }.items()
            if v is not None
        }

        transformed_images = transform_images(
            images_array, transformations, **transform_kwargs
        )

        num_images = len(images_array)
        num_rows = (
            num_images - 1
        ) // 3 + 1  # Calculate rows needed for original images
        total_rows = num_rows * 2  # Double the rows to accommodate transformed images

        plt.figure(figsize=(15, 10))
        plt.suptitle("Original vs Transformed Images")

        for i in range(num_images):
            plt.subplot(total_rows, 3, i + 1)
            plt.imshow(images_array[i])
            plt.title(f"Original {i+1}")
            plt.axis("off")

        start_idx = num_rows * 3  # Start index for transformed images
        for i in range(num_images):
            plt.subplot(total_rows, 3, start_idx + i + 1)
            plt.imshow(transformed_images[i])
            plt.title(f"Transformed {i+1}")
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
    "/normalize_image",
    responses={
        200: {
            "content": {
                "application/json": {},
            },
            "description": "Return a JSON file with the normalized image.",
        }
    },
    response_class=Response,
)
def normalize_image_endpoint(
    image: UploadFile = File(description="The image to normalize"),
):
    try:
        img = Image.open(image.file)
        img_array = np.array(img)

        normalized_image = normalize_pixel_values(img_array)

        headers = {
            "Content-Disposition": 'attachment; filename="normalized_image.json"'
        }

        normalized_image_output = {
            "normalized_image": normalized_image.tolist(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        image.file.close()

    return Response(
        json.dumps(normalized_image_output),
        headers=headers,
        media_type="application/json",
    )
