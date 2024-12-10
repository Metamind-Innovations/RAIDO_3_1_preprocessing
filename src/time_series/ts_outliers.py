import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.stats import zscore


# TODO: Add Handling for Multivariate timeseries
def detect_outliers(df: pd.DataFrame, method: str = 'all', voting_threshold: int = 2) -> pd.DataFrame:
    """
    Detect outliers in a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. The first column is expected to be a datetime column.
    method : str, default 'all'
        The outlier detection method to use. 'all' means that all methods are used and a voting system is
        used to determine the final outlier status. 'single' means that each outlier detection method is
        used separately and the union of the outliers is returned.
    voting_threshold : int, default 2
        The minimum number of outlier detection methods that must detect an outlier for it to be considered
        as an outlier.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same columns as the input DataFrame and an additional column 'Final_Outlier'
        indicating whether the row is an outlier or not.
    """
    datetime_column = df.columns[0]
    value_columns = df.columns[1:].tolist()

    # Convert first column to datetime
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')

    for column in value_columns:
        # Replace commas with dots and convert to float
        df[column] = df[column].astype(str).str.replace(',', '.').astype(float)

    # Process each value column
    for column in value_columns:
        # Prepare data for outlier detection
        values = df[column].values.reshape(-1, 1)

        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        col_name_iso = f'{column}_IsoForest'
        df[col_name_iso] = iso_forest.fit_predict(values)

        # DBSCAN
        dbscan = DBSCAN(eps=1e9, min_samples=2)
        dbscan_labels = dbscan.fit_predict(values)
        col_name_db = f'{column}_DBSCAN'
        df[col_name_db] = np.where(dbscan_labels == -1, -1, 1)

        # Z-score
        col_name_zscore = f'{column}_Zscore'
        z_scores = zscore(df[column].values)
        df[col_name_zscore] = np.where(z_scores > 3, -1, 1)

        # Voting system
        voting_columns = [f'{col}_IsoForest' for col in value_columns] + \
                         [f'{col}_DBSCAN' for col in value_columns] + \
                         [f'{col}_Zscore' for col in value_columns]

        df['Outlier_Votes'] = (df[voting_columns] == -1).sum(axis=1)
        df['Final_Outlier'] = np.where(df['Outlier_Votes'] >= voting_threshold, -1, 1)

        if method == 'all':
            return df[df['Final_Outlier'] == -1]
        elif method == 'single':
            return df[(df[[f'{col}_IsoForest' for col in value_columns]] == -1).any(axis=1) |
                      (df[[f'{col}_DBSCAN' for col in value_columns]] == -1).any(axis=1) |
                      (df[[f'{col}_Zscore' for col in value_columns]] == -1).any(axis=1)]

        else:
            raise ValueError("Invalid method. Use 'all' or 'single'.")


# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# outliers = detect_outliers(df, 'all', voting_threshold=2)
# print(f'Outliers: {outliers}')
# df.to_csv('ts_outliers.csv', index=False)
