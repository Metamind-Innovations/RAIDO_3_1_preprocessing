import json
from typing import List

from fastapi import APIRouter, File, UploadFile, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
import numpy as np
import pandas as pd
from PIL import Image

from api.models import ClassDistributionAnalysis, ClassImbalanceEvaluation, ImageSize
from src.images.balancing import (
    analyze_class_distribution,
    evaluate_class_imbalance,
    oversample_minority_classes,
    smote_oversampling,
)
from src.images.utils import resize_image


router = APIRouter(prefix="/images/balancing", tags=["Image Balancing"])


@router.post(
    "/analyze_class_distribution",
    responses={
        200: {
            "content": {
                "application/json": {},
            },
            "description": "Return a JSON file with the class distribution analysis.",
        }
    },
    response_model=ClassDistributionAnalysis,
)
def analyze_class_distribution_endpoint(
    labels: UploadFile = File(description="CSV containing the image labels"),
):
    try:
        labels_df = pd.read_csv(labels.file)
        label_list = labels_df["CLASS"].tolist()

        result = analyze_class_distribution(label_list)
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        labels.file.close()
    return ClassDistributionAnalysis(**result)


@router.post(
    "/evaluate_class_imbalance",
    responses={
        200: {
            "content": {
                "application/json": {},
            },
            "description": "Return a JSON file with the class imbalance evaluation.",
        }
    },
    response_model=ClassImbalanceEvaluation,
)
def evaluate_class_imbalance_endpoint(
    distribution_stats: UploadFile = File(
        description="JSON containing the class distribution analysis"
    ),
    class_percentage_mild_deviation: float = Query(
        10,
        description="The percentage deviation from the ideal class percentage before being considered a mild imbalance",
        gt=0,
        lt=100,
    ),
    class_percentage_severe_deviation: float = Query(
        20,
        description="The percentage deviation from the ideal class percentage before being considered a severe imbalance",
        gt=0,
        lt=100,
    ),
):
    try:
        distribution_stats_dict = json.load(distribution_stats.file)
        class_specific_imbalances = evaluate_class_imbalance(
            distribution_stats_dict,
            class_percentage_mild_deviation,
            class_percentage_severe_deviation,
        )
        result = {
            "class_specific_imbalances": class_specific_imbalances,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        distribution_stats.file.close()
    return ClassImbalanceEvaluation(**result)


@router.post(
    "/oversample_minority_classes",
    responses={
        200: {
            "content": {
                "application/json": {},
            },
            "description": "Return a JSON file containing the original images and labels, and the oversampled images and labels.",
        }
    },
    response_class=Response,
)
def oversample_minority_classes_endpoint(
    images: List[UploadFile] = File(
        description="The images to perform oversampling on"
    ),
    labels: UploadFile = File(description="CSV containing the image labels"),
    image_size: ImageSize = Query(..., description="The size of the resized images"),
):
    try:
        images_list = []
        for image in images:
            img = Image.open(image.file)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img)
            img_array = resize_image(img_array, height=image_size, width=image_size)
            images_list.append(img_array)

        labels_df = pd.read_csv(labels.file)
        labels_list = labels_df["CLASS"].tolist()

        X_oversampled, y_oversampled = oversample_minority_classes(
            images_list, labels_list
        )
        X_oversampled = [x.tolist() for x in X_oversampled]

        oversampled_images_output = {
            "X_oversampled": X_oversampled,
            "y_oversampled": y_oversampled,
        }

        headers = {
            "Content-Disposition": 'attachment; filename="oversampled_images.json"'
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        for image in images:
            image.file.close()
        labels.file.close()
    return Response(
        json.dumps(oversampled_images_output),
        headers=headers,
        media_type="application/json",
    )


@router.post(
    "/smote_oversampling",
    responses={
        200: {
            "content": {
                "application/json": {},
            },
            "description": "Return a JSON file containing the original images and labels, and the oversampled images and labels.",
        }
    },
    response_class=Response,
)
def smote_oversampling_endpoint(
    images: List[UploadFile] = File(
        description="The images to perform oversampling on"
    ),
    labels: UploadFile = File(description="CSV containing the image labels"),
    k_neighbors: int = Query(2, description="The number of neighbors to use for SMOTE"),
    image_size: ImageSize = Query(..., description="The size of the resized images"),
):
    try:
        images_list = []
        for image in images:
            img = Image.open(image.file)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img)
            img_array = resize_image(img_array, height=image_size, width=image_size)
            images_list.append(img_array)

        labels_df = pd.read_csv(labels.file)
        labels_list = labels_df["CLASS"].tolist()

        X_smote, y_smote = smote_oversampling(images_list, labels_list, k_neighbors)

        X_smote = [x.tolist() for x in X_smote]

        smote_images_output = {
            "X_smote": X_smote,
            "y_smote": y_smote,
        }

        headers = {
            "Content-Disposition": 'attachment; filename="oversampled_images.json"'
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"})
        )
    finally:
        for image in images:
            image.file.close()
        labels.file.close()
    return Response(
        json.dumps(smote_images_output),
        headers=headers,
        media_type="application/json",
    )


# TODO: Add endpoint to visualize the original and SMOTE oversampled images
