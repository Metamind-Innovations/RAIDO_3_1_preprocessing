from typing import List

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from api.models import ImputationNameTimeseries, NoiseRemovalMethod, OutlierNameTimeseries

# Missing data
from src.time_series.ts_missing_data import normalize_data, impute_missing_data, cleanup_df_zero_nans

# Outliers
from src.time_series.ts_outliers import detect_outliers

# Noise
from src.time_series.noise import ema, fourier_transform, savitzky_golay, wavelet_denoising

# Feature Engineering
from src.time_series.feature_engineering import extract_date_features, calculate_differences, \
    one_hot_encode_categoricals

router = APIRouter(prefix="/time_series", tags=["Time Series"])


def cleanup(df):
    for column in df.columns[1:]:
        df[column] = pd.to_numeric(df[column].astype(str).str.replace(',', '.'), errors='coerce')
    df = cleanup_df_zero_nans(df)
    return df


# # # TODOS # # #
# TODO: Later the file will be loaded from a url. Add async await to all endpoints


@router.post("/missing_data/impute")
def impute_missing_data_endpoint(csv: UploadFile = File(description="The csv to impute missing data"),
                                 method: ImputationNameTimeseries = Query(...,
                                                                          description="The method to use for imputation")):
    try:
        # Read csv with proper parsing of dates and separator
        df = pd.read_csv(csv.file, sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
        df = cleanup(df)
        imputed_df = impute_missing_data(df, method)
        normalized_df = normalize_data(imputed_df)
        return normalized_df.to_dict(orient='records')
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/outliers/detect")
def detect_outliers_endpoint(
        csv: UploadFile = File(description="The csv to check for outliers"),
        method: OutlierNameTimeseries = Query(OutlierNameTimeseries.all,
                                              description="The method to use for outlier detection"),
        voting_threshold: int = Query(2,
                                      description="The minimum number of outlier detection methods that must detect an outlier for it to be considered as an outlier."),
):
    try:
        # Read csv with proper parsing of dates and separator
        df = pd.read_csv(csv.file, sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
        outliers_df = detect_outliers(df, method, voting_threshold=voting_threshold)
        return outliers_df.to_dict(orient='records')
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/noise_removal")
def noise_removal_endpoint(
        csv: UploadFile = File(description="The csv to remove noise"),
        method: NoiseRemovalMethod = Query(
            ...,
            description="Method used for noise removal. Possible values: 'ema', 'fourier', 'savitzky', 'wavelet'"
        ),
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        denoising_methods = {
            NoiseRemovalMethod.ema: ema,
            NoiseRemovalMethod.fourier: fourier_transform,
            NoiseRemovalMethod.savitzky: savitzky_golay,
            NoiseRemovalMethod.wavelet: wavelet_denoising,
        }
        denoised_df = denoising_methods[method](df)
        return denoised_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/feature_engineering/date_features")
def feature_engineering_date_features_endpoint(
        csv: UploadFile = File(description="The csv to engineer date features"),
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        engineered_df = extract_date_features(df)
        return engineered_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/feature_engineering/difference")
def feature_engineering_difference_endpoint(
        csv: UploadFile = File(description="The csv to engineer differences"),
        column: str = Query(default="value", description="The column to calculate the differences for"),
        order: int = Query(default=1, description="The order of the differences"),
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        engineered_df = calculate_differences(df, column, order)
        return engineered_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/feature_engineering/one_hot")
def feature_engineering_one_hot_endpoint(
        csv: UploadFile = File(description="The csv to engineer one-hot features"),
        list_columns: List[str] = Query(default=["value"], description="The list of columns to one-hot encode"),
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        engineered_df = one_hot_encode_categoricals(df, list_columns)
        return engineered_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()
