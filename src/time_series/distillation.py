import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


def top_k_distillation(df, column, k=10):
    """
    Select the top-K highest values from the dataset.

    :param df: The input DataFrame containing time series data.
    :type df: pd.DataFrame
    :param k: The number of top values to select, defaults to 5.
    :type k: int, optional
    :return: A DataFrame containing the top-K highest values.
    :rtype: pd.DataFrame
    """
    top_k_df = df.nlargest(k, column).sort_values(df.columns[0]).reset_index(drop=True)
    return top_k_df


def threshold_based_distillation(df, column, threshold=10000):
    """
    Retain only the values that are above a certain threshold.

    :param df: The input DataFrame containing time series data.
    :type df: pd.DataFrame
    :param threshold: The value threshold, defaults to 10000.
    :type threshold: int, optional
    :return: A DataFrame containing values above the threshold.
    :rtype: pd.DataFrame
    """
    threshold_df = df[df[column] > threshold].reset_index(drop=True)
    return threshold_df


def tf_based_median_distillation(df, timeframe='d'):
    """
    Compute the median value for each specified timeframe in the dataset.

    :param df: The input DataFrame containing time series data.
    :type df: pd.DataFrame
    :param timeframe: The timeframe to resample the data, defaults to 'd' (day).
    :type timeframe: str, optional
    :return: A DataFrame containing the median values for each specified timeframe.
    :rtype: pd.DataFrame
    """
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df.set_index(df.columns[0], inplace=True)
    median_df = df.resample(timeframe).median().reset_index()
    median_df.fillna(0.0, inplace=True)
    return median_df


def percentile_based_distillation(df, column, lower_percentile=0.25, upper_percentile=0.75):
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
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    percentile_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].reset_index(drop=True)
    return percentile_df


def peak_detection_distillation(df, column, height=10000, distance=5):
    """
    Identify and retain peak values in the time series.

    :param df: The input DataFrame containing time series data.
    :type df: pd.DataFrame
    :param height: The required height of peaks, defaults to 10000.
    :type height: float, optional
    :param distance: The required minimal horizontal distance (in samples) between neighboring peaks, defaults to 5.
    :type distance: float, optional
    :return: A DataFrame containing the peak values.
    :rtype: pd.DataFrame
    """
    peaks, _ = find_peaks(df[column], height=height, distance=distance)
    peak_df = df.iloc[peaks].reset_index(drop=True)
    return peak_df


def step_distill(df, feedback_steps=5):
    """
    Implement DynaDistill method for dataset distillation.
    :param df: The input DataFrame containing time series data.
    :type df: pd.DataFrame
    :param column: The column to distill.
    :type column: str
    :param feedback_steps: Number of feedback steps, defaults to 5.
    :type feedback_steps: int, optional
    :return: A DataFrame containing distilled data.
    :rtype: pd.DataFrame
    """
    # Select every nth point as a simple feedback mechanism
    distilled_df = df.iloc[::feedback_steps].reset_index(drop=True)
    return distilled_df


def clustering_based_distillation(df, column, n_clusters=10):
    """
    Implement a clustering-based distillation method for time series data.
    :param df: The input DataFrame containing time series data.
    :type df: pd.DataFrame
    :param column: The column to distill.
    :type column: str
    :param n_clusters: Number of clusters to form, defaults to 5.
    :type n_clusters: int, optional
    :return: A DataFrame containing the cluster centroids.
    :rtype: pd.DataFrame
    """
    # Reshape the data for clustering
    data = df[column].values.reshape(-1, 1)
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    # Get the cluster labels
    labels = kmeans.labels_
    # Get the cluster centroids
    centroids = kmeans.cluster_centers_.flatten()
    # Create a new DataFrame with the centroids
    df[f'{column}_centroid'] = [centroids[label] for label in labels]

    return df

# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# top_k_df = top_k_distillation(df, k=10)
# print(top_k_df)
#
# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# threshold_df = threshold_based_distillation(df, threshold=10000)
# print(threshold_df)
#
# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# median_df = tf_based_median_distillation(df, 'd')
# print(median_df)
#
# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# percentile_df = percentile_based_distillation(df, lower_percentile=0.25, upper_percentile=0.75)
# print(percentile_df)
#
# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# peak_df = peak_detection_distillation(df, height=10000, distance=5)
# print(peak_df)

# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# distilled_df = step_distill(df, feedback_steps=5)
# print(distilled_df)

# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# centroids_df = clustering_based_distillation(df, 'value')
# print(centroids_df.head(50))
