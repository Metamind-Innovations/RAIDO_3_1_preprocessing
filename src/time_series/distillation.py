import pandas as pd
from scipy.signal import find_peaks


def top_k_distillation(df, k=5):
    """
    Select the top-K highest values from the dataset.

    :param df: The input DataFrame containing time series data.
    :type df: pd.DataFrame
    :param k: The number of top values to select, defaults to 5.
    :type k: int, optional
    :return: A DataFrame containing the top-K highest values.
    :rtype: pd.DataFrame
    """
    top_k_df = df.nlargest(k, 'value').sort_values('time').reset_index(drop=True)
    return top_k_df


def threshold_based_distillation(df, threshold=10000):
    """
    Retain only the values that are above a certain threshold.

    :param df: The input DataFrame containing time series data.
    :type df: pd.DataFrame
    :param threshold: The value threshold, defaults to 10000.
    :type threshold: int, optional
    :return: A DataFrame containing values above the threshold.
    :rtype: pd.DataFrame
    """
    threshold_df = df[df['value'] > threshold].reset_index(drop=True)
    return threshold_df


def daily_median_distillation(df):
    """
    Compute the median value for each day in the dataset.

    :param df: The input DataFrame containing time series data.
    :type df: pd.DataFrame
    :return: A DataFrame containing the daily median values.
    :rtype: pd.DataFrame
    """
    df['date'] = df['time'].dt.date
    median_df = df.groupby('date')['value'].median().reset_index()
    median_df['time'] = pd.to_datetime(median_df['date'])
    median_df = median_df[['time', 'value']]
    return median_df


def percentile_based_distillation(df, lower_percentile=0.25, upper_percentile=0.75):
    """
    Retain values within a specified percentile range.

    :param df: The input DataFrame containing time series data.
    :type df: pd.DataFrame
    :param lower_percentile: The lower percentile boundary, defaults to 0.25.
    :type lower_percentile: float, optional
    :param upper_percentile: The upper percentile boundary, defaults to 0.75.
    :type upper_percentile: float, optional
    :return: A DataFrame containing values within the specified percentile range.
    :rtype: pd.DataFrame
    """
    lower_bound = df['value'].quantile(lower_percentile)
    upper_bound = df['value'].quantile(upper_percentile)
    percentile_df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)].reset_index(drop=True)
    return percentile_df


def peak_detection_distillation(df, height=None, distance=None):
    """
    Identify and retain peak values in the time series.

    :param df: The input DataFrame containing time series data.
    :type df: pd.DataFrame
    :param height: The required height of peaks, defaults to None.
    :type height: float, optional
    :param distance: The required minimal horizontal distance (in samples) between neighboring peaks, defaults to None.
    :type distance: float, optional
    :return: A DataFrame containing the peak values.
    :rtype: pd.DataFrame
    """
    peaks, _ = find_peaks(df['value'], height=height, distance=distance)
    peak_df = df.iloc[peaks].reset_index(drop=True)
    return peak_df


# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# top_k_df = top_k_distillation(df, k=10)
# print(top_k_df)
#
# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# threshold_df = threshold_based_distillation(df, threshold=10000)
# print(threshold_df)
#
# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# median_df = daily_median_distillation(df)
# print(median_df)
#
# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# percentile_df = percentile_based_distillation(df, lower_percentile=0.25, upper_percentile=0.75)
# print(percentile_df)
#
# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# peak_df = peak_detection_distillation(df, height=10000, distance=5)
# print(peak_df)
