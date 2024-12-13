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

# Dim Reduction
from src.time_series.dim_reduction import pca_dim_reduction, isometric_mapping, autoencoder_reduction, pvqa

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


@router.post("/dimensionality_reduction/pca")
def dimensionality_reduction_pca_endpoint(
        csv: UploadFile = File(description="The csv to perform PCA on"),
        column: str = Query(default="value", description="The column to perform PCA on"),
        max_components: int = Query(default=None, description="The maximum number of components to keep"),
):
    """
    Apply Principal Component Analysis (PCA) to a given dataset of time series data.

    Parameters
    ----------
    csv : UploadFile
        The csv to perform PCA on.
    column : str
        The column to perform PCA on.
    max_components : int
        The maximum number of components to keep.

    Returns
    -------
    JSONResponse
        A JSON response containing the reduced data.
    """
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        reduced_df = pca_dim_reduction(df, column, max_components)
        return reduced_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/dimensionality_reduction/isomap")
def dimensionality_reduction_isomap_endpoint(
        csv: UploadFile = File(description="The csv to perform Isomap on"),
        n_components: int = Query(default=2, description="The number of components to keep"),
        column: str = Query(default="value", description="The column to perform Isomap on"),
        n_neighbors: int = Query(default=10, description="The number of neighbors to consider"),
):
    """
    Apply Isometric Mapping (Isomap) to a given dataset of time series data.

    Parameters
    ----------
    csv : UploadFile
        The csv to perform Isomap on.
    n_components : int
        The number of components to keep.
    column : str
        The column to perform Isomap on.
    n_neighbors : int
        The number of neighbors to consider.

    Returns
    -------
    JSONResponse
        A JSON response containing the reduced data.
    """
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        reduced_df = isometric_mapping(df, n_components, column, n_neighbors)
        return reduced_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/dimensionality_reduction/pvqa")
def dimensionality_reduction_pvqa_endpoint(
        csv: UploadFile = File(description="The csv to perform PVQA on"),
        num_segments: int = Query(default=10, description="The number of segments to divide the data into"),
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        reduced_df = pvqa(df, num_segments)
        return reduced_df.to_dict(orient="records")
    except pd.errors.EmptyDataError:
        return JSONResponse(status_code=400, content={"message": "The CSV file is empty."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})
    finally:
        csv.file.close()


@router.post("/dimensionality_reduction/autoencoder")
def dimensionality_reduction_autoencoder_endpoint(
        csv: UploadFile = File(description="The csv to perform autoencoder on"),
        column: str = Query(default="value", description="The column to perform autoencoder on"),
        n_lag: int = Query(default=2, description="The number of components to keep"),
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        reduced_df = autoencoder_reduction(df, column, n_lag)
        return reduced_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()
