from typing import List
import numpy as np
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

# Balancing
from src.time_series.balancing import upsampling, downsampling, rolling_window

# Enrichment
from src.time_series.enrichment import enrich_with_statistics, enrich_with_temporal_features, \
    enrich_with_anomaly_detection, add_polynomial_features, add_log_features, add_cyclical_features, standardize_data

# Distillation
from src.time_series.distillation import threshold_based_distillation, tf_based_median_distillation, \
    percentile_based_distillation, peak_detection_distillation, top_k_distillation, step_distill, \
    clustering_based_distillation

router = APIRouter(prefix="/time_series", tags=["Time Series"])


def cleanup(df):
    for column in df.columns[1:]:
        df[column] = pd.to_numeric(df[column].astype(str).str.replace(',', '.'), errors='coerce')
    df = cleanup_df_zero_nans(df)
    return df


# # # TODOS # # #
# TODO: Later the file will be loaded from a url. Add async await to all endpoints


@router.get("/endpoints")
def list_endpoints():
    try:
        endpoints = []
        for route in router.routes:
            if hasattr(route, "methods"):
                for method in route.methods:
                    if method in ["GET", "POST", "PUT", "DELETE"]:
                        endpoints.append({
                            "path": route.path,
                            "method": method,
                            "name": route.name
                        })
        return JSONResponse(status_code=200, content=jsonable_encoder(endpoints))
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))


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


# TODO: This works in prototype but not in FastAPI. The endpoint hangs.
@router.post("/balancing/upsample")
def balancing_upsample_endpoint(
        csv: UploadFile = File(description="The csv to upsample"),
        target_frequency: str = Query(default='min', description="The target frequency for upsampling"),
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        upsampled_df = upsampling(df, target_frequency)
        return upsampled_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/balancing/downsample")
def balancing_downsample_endpoint(
        csv: UploadFile = File(description="The csv to downsample"),
        target_frequency: str = Query(default='h', description="The target frequency for downsampling"),
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        downsampled_df = downsampling(df, target_frequency)
        return downsampled_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/balancing/rolling_window")
def balancing_rolling_window_endpoint(
        csv: UploadFile = File(description="The csv to apply the rolling window"),
        window_size: int = Query(default=2, description="The size of the rolling window"),
        target_columns: List[str] = Query(default=None, description="List of column names to apply rolling window"),
        aggregation_method: str = Query(default='mean',
                                        description="Method to aggregate values ('mean', 'sum', 'std', 'min', 'max')"),
        min_periods: int = Query(default=1, description="Minimum number of observations required for calculation"),
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        rolled_df = rolling_window(df, window_size, target_columns, aggregation_method, min_periods)
        return rolled_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/enrichment/statistics")
def enrichment_statistics_endpoint(
        csv: UploadFile = File(description="The csv to enrich with statistics"),
        column: str = Query(default="value", description="The column to calculate statistics for"),
        window_sizes: List[int] = Query(default=[5],
                                        description="List of window sizes for rolling calculations"),
        quantiles: List[float] = Query(default=[0.25],
                                       description="List of quantiles to calculate for each window size")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True)
        enriched_df = enrich_with_statistics(df, column, window_sizes, quantiles)
        enriched_df = enriched_df.where(pd.notnull(enriched_df), 0)
        return enriched_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})
    finally:
        csv.file.close()


@router.post("/enrichment/temporal_features")
def enrichment_temporal_features_endpoint(
        csv: UploadFile = File(description="The csv to enrich with temporal features"),
        column: str = Query(default="value", description="The column to calculate temporal features for")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        enriched_df = enrich_with_temporal_features(df, column)
        enriched_df = enriched_df.where(pd.notnull(enriched_df), 0)
        return enriched_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/enrichment/anomaly_detection")
def enrichment_anomaly_detection_endpoint(
        csv: UploadFile = File(description="The csv to enrich with anomaly detection"),
        column: str = Query(default="value", description="The column to detect anomalies in"),
        contamination: float = Query(default=0.01,
                                     description="The proportion of anomalies in the data for isolation forest")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        enriched_df = enrich_with_anomaly_detection(df, column, contamination)
        enriched_df = enriched_df.replace([np.inf, -np.inf], np.nan)
        enriched_df = enriched_df.where(pd.notnull(enriched_df), 0)
        return enriched_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/enrichment/polynomial_features")
def enrichment_polynomial_features_endpoint(
        csv: UploadFile = File(description="The csv to enrich with polynomial features"),
        column: str = Query(default="value", description="The column to create polynomial features for"),
        degree: int = Query(default=2, description="The degree of the polynomial features")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        enriched_df = add_polynomial_features(df, column, degree)
        enriched_df = enriched_df.where(pd.notnull(enriched_df), 0)
        return enriched_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/enrichment/log_features")
def enrichment_log_features_endpoint(
        csv: UploadFile = File(description="The csv to enrich with log features"),
        column: str = Query(default="value", description="The column to create log features for")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        enriched_df = add_log_features(df, column)
        # Remove inf values
        enriched_df = enriched_df.replace([np.inf, -np.inf], np.nan)
        enriched_df = enriched_df.where(pd.notnull(enriched_df), 0)
        return enriched_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/enrichment/cyclical_features")
def enrichment_cyclical_features_endpoint(
        csv: UploadFile = File(description="The csv to enrich with cyclical features"),
        column: str = Query(default="value", description="The column to create cyclical features for"),
        period: int = Query(default=24, description="The period for the cyclical features")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        enriched_df = add_cyclical_features(df, column, period)
        return enriched_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/standardize_data")
def standardize_data_endpoint(
        csv: UploadFile = File(description="The csv to standardize"),
        column: str = Query(default="value", description="The column to standardize")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        standardized_df = standardize_data(df, column)
        return standardized_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/distillation/top_k")
def distillation_top_k_endpoint(
        csv: UploadFile = File(description="The csv to apply top-k distillation"),
        column: str = Query(default="value", description="The column to select top-k values from"),
        k: int = Query(default=10, description="The number of top values to select")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        distilled_df = top_k_distillation(df, column, k)
        return distilled_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/distillation/threshold")
def distillation_threshold_endpoint(
        csv: UploadFile = File(description="The csv to apply threshold-based distillation"),
        column: str = Query(default="value", description="The column to apply threshold on"),
        threshold: int = Query(default=10000, description="The value threshold")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        distilled_df = threshold_based_distillation(df, column, threshold)
        return distilled_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/distillation/tf_based_median")
def tf_based_median_distillation_endpoint(
        csv: UploadFile = File(description="The csv to apply daily median distillation"),
        timeframe: str = Query(default='d',
                               description="The timeframe on which to calculate median. "
                                           "Some possible values are: d (day), h (hour), min (minute)")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        distilled_df = tf_based_median_distillation(df, timeframe)
        return distilled_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/distillation/percentile")
def distillation_percentile_endpoint(
        csv: UploadFile = File(description="The csv to apply percentile-based distillation"),
        column: str = Query(default="value", description="The column to apply percentile-based distillation on"),
        lower_percentile: float = Query(default=0.25, description="The lower percentile boundary"),
        upper_percentile: float = Query(default=0.75, description="The upper percentile boundary")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        distilled_df = percentile_based_distillation(df, column, lower_percentile, upper_percentile)
        return distilled_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/distillation/peak_detection")
def distillation_peak_detection_endpoint(
        csv: UploadFile = File(description="The csv to apply peak detection distillation"),
        column: str = Query(default="value", description="The column to detect peaks in"),
        height: float = Query(default=10000, description="The required height of peaks"),
        distance: int = Query(default=5, description="The required minimal horizontal distance between peaks")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        distilled_df = peak_detection_distillation(df, column, height, distance)
        return distilled_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/distillation/step")
def distillation_step_endpoint(
        csv: UploadFile = File(description="The csv to apply step distillation"),
        feedback_steps: int = Query(default=5, description="The number of feedback steps")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        distilled_df = step_distill(df, feedback_steps)
        return distilled_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()


@router.post("/distillation/clustering")
def distillation_clustering_endpoint(
        csv: UploadFile = File(description="The csv to apply clustering-based distillation"),
        column: str = Query(default="value", description="The column to distill"),
        n_clusters: int = Query(default=10, description="The number of clusters to form")
):
    try:
        df = pd.read_csv(csv.file, sep=";", parse_dates=[0], dayfirst=True, low_memory=False)
        distilled_df = clustering_based_distillation(df, column, n_clusters)
        return distilled_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content=jsonable_encoder({"message": f"Error: {str(e)}"}))
    finally:
        csv.file.close()
