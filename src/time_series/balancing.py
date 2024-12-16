import pandas as pd


def upsampling(df: pd.DataFrame, target_frequency: str = 'min') -> pd.DataFrame:
    """
    Upsample the dataset to a specified frequency using linear interpolation.

    :param df: The DataFrame containing the time series data.
    :type df: pd.DataFrame
    :param target_frequency: The target frequency for upsampling. Default is 'min' (minute).
        Possible values are:
        - 's': second
        - 't': minute
        - 'h': hour
        - 'd': day
        - 'm': month
        - 'y': year
    :type target_frequency: str, optional
    :return: The upsampled DataFrame.
    :rtype: pd.DataFrame
    """
    df_upsampled = df.set_index(df.columns[0]).resample(target_frequency).interpolate('linear').reset_index()
    return df_upsampled


def downsampling(df: pd.DataFrame, target_frequency: str = 'h') -> pd.DataFrame:
    """
    Downsample the dataset to a specified frequency using mean aggregation.

    :param df: The DataFrame containing the time series data.
    :type df: pd.DataFrame
    :param target_frequency: The target frequency for downsampling. Default is 'h' (hour).
        Possible values are:
        - 's': second
        - 't': minute
        - 'h': hour
        - 'd': day
        - 'm': month
        - 'y': year
    :type target_frequency: str, optional
    :return: The downsampled DataFrame.
    :rtype: pd.DataFrame
    """
    df_downsampled = df.set_index(df.columns[0]).resample(target_frequency).mean().dropna().reset_index()
    return df_downsampled

# TODO: Add SMOTE for classification problems


def rolling_window(
        df: pd.DataFrame,
        window_size: int = 2,
        target_columns: list = None,
        aggregation_method: str = 'mean',
        min_periods: int = 1
) -> pd.DataFrame:
    """
    Apply a rolling window to create sequences of data points with various aggregation methods.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the time series data
    window_size : int, optional
        The size of the rolling window (default: 3)
    target_columns : list, optional
        List of column names to apply rolling window (default: all numeric columns)
    aggregation_method : str, optional
        Method to aggregate values ('mean', 'sum', 'std', 'min', 'max') (default: 'mean')
    min_periods : int, optional
        Minimum number of observations required for calculation (default: 1)

    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling window sequences
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    df_copy = df.copy()

    # If no target columns specified, use all numeric columns
    if target_columns is None:
        target_columns = df_copy.select_dtypes(include=['int64', 'float64']).columns

    # Validate target columns exist in DataFrame
    if not all(col in df_copy.columns for col in target_columns):
        raise ValueError("One or more target columns not found in DataFrame")

    # Aggregation methods
    agg_methods = {
        'mean': lambda x: x.rolling(window=window_size, min_periods=min_periods).mean(),
        'sum': lambda x: x.rolling(window=window_size, min_periods=min_periods).sum(),
        'std': lambda x: x.rolling(window=window_size, min_periods=min_periods).std(),
        'min': lambda x: x.rolling(window=window_size, min_periods=min_periods).min(),
        'max': lambda x: x.rolling(window=window_size, min_periods=min_periods).max()
    }

    if aggregation_method not in agg_methods:
        raise ValueError(f"Unsupported aggregation method. Choose from {list(agg_methods.keys())}")

    # Apply rolling window to each target column
    for col in target_columns:
        # Create lagged features
        for i in range(1, window_size):
            df_copy[f'{col}_lag_{i}'] = df_copy[col].shift(i)

        # Apply aggregation method
        df_copy[f'{col}_{aggregation_method}_{window_size}'] = agg_methods[aggregation_method](df_copy[col])

    # Drop rows with NaN values and reset index
    df_copy = df_copy.dropna().reset_index(drop=True)

    return df_copy


# if __name__ == "__main__":
    # df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True)
    #
    # print(df.head(30))
    # # Upsample
    # df_upsampled = upsampling(df)
    # print("Upsampled DataFrame:")
    # print(df_upsampled.head(30))
    #
    # # Downsample
    # df_downsampled = downsampling(df)
    # print("Downsampled DataFrame:")
    # print(df_downsampled.head(30))
    #
    # # Rolling Window
    # df_rolling = rolling_window(df)
    # print("Rolling Window DataFrame:")
    # print(df_rolling.head(30))