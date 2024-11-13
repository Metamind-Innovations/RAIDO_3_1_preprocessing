import io
import json
from PIL import Image

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from matplotlib import pyplot as plt
import numpy as np

from api.models import DenoiseMode
from src.images.noise import denoise_non_local_means


router = APIRouter(prefix="/images/noise_removal", tags=["Images Noise Removal"])


@router.post(
    "/denoise_non_local_means",
    responses={
        200: {
            "content": {
                "application/json": {},
            },
            "description": "Return a JSON file with the denoised image.",
        }
    },
    response_class=Response,
)
def denoise_non_local_means_endpoint(
    image: UploadFile = File(description="The noisy image"),
    patch_size: int = Query(7, description="The size of the patches to extract"),
    patch_distance: int = Query(11, description="The distance between patches"),
    mode: DenoiseMode = Query(
        DenoiseMode.fast, description="The mode to use for denoising (slow/fast)"
    ),
):
    try:
        img = Image.open(image.file)
        img_array = np.array(img)

        if mode is DenoiseMode.fast:
            fast_mode = True
        else:
            fast_mode = False

        if img.mode == "RGB" or img.mode == "RGBA":
            multichannel = True
        else:
            multichannel = False

        denoised_image = denoise_non_local_means(
            img_array, patch_size, patch_distance, fast_mode, multichannel
        )

        headers = {"Content-Disposition": 'attachment; filename="denoised_image.json"'}

        denoise_non_local_means_output = {"denoised_image": denoised_image.tolist()}
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        image.file.close()

    return Response(
        json.dumps(denoise_non_local_means_output),
        headers=headers,
        media_type="application/json",
    )


@router.post(
    "/visualize_denoised_image",
    responses={
        200: {
            "content": {
                "image/png": {},
            },
            "description": "Return a PNG file with the plots of the original and denoised image.",
        }
    },
    response_class=Response,
)
def visualize_denoised_image_endpoint(
    image: UploadFile = File(description="The noisy image"),
    patch_size: int = Query(7, description="The size of the patches to extract"),
    patch_distance: int = Query(11, description="The distance between patches"),
    mode: DenoiseMode = Query(
        DenoiseMode.fast, description="The mode to use for denoising (slow/fast)"
    ),
):
    try:
        img = Image.open(image.file)
        img_array = np.array(img)

        if mode is DenoiseMode.fast:
            fast_mode = True
        else:
            fast_mode = False

        if img.mode == "RGB" or img.mode == "RGBA":
            multichannel = True
        else:
            multichannel = False

        denoised_image = denoise_non_local_means(
            img_array, patch_size, patch_distance, fast_mode, multichannel
        )

        plt.figure(figsize=(12, 6))
        plt.suptitle("Non-Local Means Denoising")

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title("Original Image")
        plt.axis("off")

        # Plot denoised image
        plt.subplot(1, 2, 2)
        plt.imshow(denoised_image)
        plt.title("Denoised Image")
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
