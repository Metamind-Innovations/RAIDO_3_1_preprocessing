import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import IsolationForest


def enrich_with_statistics(df, column='value', window_sizes=[5, 10, 20], quantiles=[0.25, 0.75]):
    """
    Add rolling statistics to the DataFrame.
    :param df: The DataFrame containing the time series data.
    :type df: pd.DataFrame
    :param column: The name of the column to calculate statistics for.
    :type column: str, optional
    :param window_sizes: List of window sizes for rolling calculations.
    :type window_sizes: List[int], optional
    :param quantiles: List of quantiles to calculate for each window size.
    :type quantiles: List[float], optional
    :return: DataFrame enriched with statistical features.
    :rtype: pd.DataFrame
    """
    # Add statistics to the DataFrame
    df[df.columns[0]] = pd.to_datetime(df['time'], format='%d/%m/%Y %H:%M')

    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df[column].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[column].rolling(window=window).std()
        df[f'rolling_max_{window}'] = df[column].rolling(window=window).max()
        df[f'rolling_min_{window}'] = df[column].rolling(window=window).min()
        df[f'rolling_median_{window}'] = df[column].rolling(window=window).median()

        for q in quantiles:
            df[f'rolling_quantile_{q}_{window}'] = df[column].rolling(window=window).quantile(q)

        df[f'rolling_skew_{window}'] = df[column].rolling(window=window).skew()
        df[f'rolling_kurtosis_{window}'] = df[column].rolling(window=window).kurt()

    df['ewma'] = df[column].ewm(span=5).mean()
    df['ewmstd'] = df[column].ewm(span=5).std()

    return df


def enrich_with_temporal_features(df, column='value'):
    """
    Add temporal features to the DataFrame.
    :param df: The DataFrame containing the time series data.
    :type df: pd.DataFrame
    :param column: The name of the column to calculate temporal features for.
    :type column: str, optional
    :return: DataFrame enriched with temporal features.
    :rtype: pd.DataFrame
    """
    # Add temporal features
    time_column = df.columns[0]
    df['time'] = pd.to_datetime(df[time_column], format='%d/%m/%Y %H:%M')
    df['hour'] = df[time_column].dt.hour
    df['minute'] = df[time_column].dt.minute
    df['day_of_week'] = df[time_column].dt.dayofweek
    df['is_weekend'] = df[time_column].dt.dayofweek.isin([5, 6]).astype(int)
    df['day_of_year'] = df[time_column].dt.dayofyear
    df['week_of_year'] = df[time_column].dt.isocalendar().week
    df['month'] = df[time_column].dt.month
    df['quarter'] = df[time_column].dt.quarter
    df['is_month_start'] = df[time_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[time_column].dt.is_month_end.astype(int)

    # Add cyclical features
    df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
    df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
    df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))

    # Add time series decomposition
    decomposition = seasonal_decompose(df[column], model='additive', period=24)
    df['trend'] = decomposition.trend
    df['seasonal'] = decomposition.seasonal
    df['residual'] = decomposition.resid

    return df


def enrich_with_anomaly_detection(df, column='value', contamination=0.01):
    """
    Add anomaly detection features to the DataFrame.
    :param df: The DataFrame containing the time series data.
    :type df: pd.DataFrame
    :param column: The name of the column to detect anomalies in.
    :type column: str, optional
    :param contamination: The proportion of anomalies in the data for isolation forest.
    :type contamination: float, optional
    :return: DataFrame enriched with anomaly detection features.
    :rtype: pd.DataFrame
    """
    # Add anomaly detection features
    scaler = StandardScaler()
    df['value_scaled'] = scaler.fit_transform(df[[column]])

    # Z-score method
    mean = df[column].mean()
    std = df[column].std()
    df['z_score'] = (df[column] - mean) / std
    df['is_zscore_anomaly'] = (abs(df['z_score']) > 3).astype(int)
    df['distance_from_mean'] = abs(df[column] - mean)

    # IQR method
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df['is_iqr_anomaly'] = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))).astype(int)

    # Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df['is_iforest_anomaly'] = iso_forest.fit_predict(df[[f'{column}_scaled']])
    df['is_iforest_anomaly'] = (df['is_iforest_anomaly'] == -1).astype(int)

    # Modified Z-score
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    modified_z_score = 0.6745 * (df[column] - median) / mad
    df['modified_z_score'] = modified_z_score
    df['is_modified_zscore_anomaly'] = (abs(modified_z_score) > 3.5).astype(int)

    # Combine anomaly detection methods
    df['anomaly_score'] = df['is_zscore_anomaly'] + df['is_iqr_anomaly'] + df['is_iforest_anomaly'] + df[
        'is_modified_zscore_anomaly']

    return df


def add_polynomial_features(df, column='value', degree=2):
    """
    Add polynomial features to the DataFrame.

    :param df: The DataFrame containing the time series data.
    :type df: pd.DataFrame
    :param column: The name of the column to create polynomial features for.
    :type column: str, optional
    :param degree: The degree of the polynomial features.
    :type degree: int, optional
    :return: DataFrame enriched with polynomial features.
    :rtype: pd.DataFrame
    """
    poly = PolynomialFeatures(degree=degree)
    poly_features = poly.fit_transform(df[[column]])
    poly_feature_names = poly.get_feature_names_out([column])

    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    df = pd.concat([df.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)

    return df


def add_log_features(df, column='value'):
    """
    Add log-transformed features to the DataFrame.

    :param df: The DataFrame containing the time series data.
    :type df: pd.DataFrame
    :param column: The name of the column to create log features for.
    :type column: str, optional
    :return: DataFrame enriched with log-transformed features.
    :rtype: pd
    :rtype: pd.DataFrame
    """
    df[f'log_{column}'] = np.log1p(df[column])
    return df


def add_cyclical_features(df, column, period):
    """
    Add cyclical encoding for a given periodic feature.

    :param df: The DataFrame containing the time series data.
    :type df: pd.DataFrame
    :param column: The name of the column to create cyclical features for.
    :type column: str
    :param period: The period of the cyclical feature (e.g., 24 for hours in a day, 7 for days in a week).
    :type period: int
    :return: DataFrame enriched with cyclical features.
    :rtype: pd.DataFrame
    """
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / period)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / period)
    return df


def standardize_data(df, column='value'):
    """
    Standardize the DataFrame the StandardScaler.

    :param df: The DataFrame containing the time series data.
    :type df: pd.DataFrame
    :param column: The name of the column to standardize.
    :type column: str, optional
    :return: DataFrame standardized using Z-score normalization.
    :rtype: pd.DataFrame
    """
    scaler = StandardScaler()
    df[f'standardized_{column}'] = scaler.fit_transform(df[[column]])
    return df


# # Example usage
# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# enriched_df = enrich_with_statistics(df)
# print(f'Enriched DataFrame (statistics): {enriched_df.head(20)}')
# enriched_df = enrich_with_temporal_features(df)
# print(f'Enriched DataFrame (temporal features): {enriched_df.head(20)}')
# enriched_df = enrich_with_anomaly_detection(df)
# print(f'Enriched DataFrame (anomaly detection): {enriched_df.head(20)}')
# enriched_df = add_polynomial_features(df)
# print(f'Enriched DataFrame (polynomial features): {enriched_df.head(20)}')
# enriched_df = add_log_features(df)
# print(f'Enriched DataFrame (log features): {enriched_df.head(20)}')
# enriched_df = add_cyclical_features(df, 'hour', 24)
# print(f'Enriched DataFrame (cyclical features): {enriched_df.head(20)}')
