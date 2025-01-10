from fastapi import APIRouter, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import pandas as pd

from api.models import ClassDistributionAnalysis
from src.images.balancing import analyze_class_distribution


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
