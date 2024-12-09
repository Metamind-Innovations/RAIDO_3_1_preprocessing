import pandas as pd
from fastapi import APIRouter, UploadFile, File, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from api.models import ImputationNameTimeseries
from src.time_series.ts_missing_data import (
    normalize_data,
    impute_missing_data,
    cleanup_df_zero_nans
)

router = APIRouter(prefix="/time_series", tags=["Time Series Missing Data"])


@router.post("/impute_missing_data")
def impute_missing_data_endpoint(csv: UploadFile = File(description="The csv to check for missing data"),
                                 method: ImputationNameTimeseries = Query(..., description="The method to use for imputation")):
    try:
        # Read csv with proper parsing of dates and separator
        df = pd.read_csv(csv.file, sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
        for column in df.columns[1:]:
            df[column] = pd.to_numeric(df[column].astype(str).str.replace(',', '.'), errors='coerce')
        df = cleanup_df_zero_nans(df)
        imputed_df = impute_missing_data(df, method)
        normalized_df = normalize_data(imputed_df)
        return normalized_df.to_dict(orient='records')
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()
