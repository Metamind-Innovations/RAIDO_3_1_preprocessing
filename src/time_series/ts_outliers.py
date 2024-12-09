import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN


# TODO: Add z-score outlier detection
# TODO: Add Handling for Multivariate timeseries
# TODO: Add voting system (if 3/4 methods say outlier, then mark it as an outlier)
def detect_outliers(df: pd.DataFrame, method: str = 'both') -> pd.DataFrame:
    """
    Detect outliers in a time series DataFrame using Isolation Forest and DBSCAN.

    Parameters
    ----------
    df : pd.DataFrame
        Time series DataFrame with datetime in the first column and values in the remaining columns.
    method : str, optional
        Method to use for outlier detection. Options are 'both' (default) for using both Isolation Forest and DBSCAN,
        or 'single' for using only one of the methods.

    Returns
    -------
    pd.DataFrame
        Outlier DataFrame with the same columns as the input DataFrame, but with additional columns for the outlier
        labels from Isolation Forest and DBSCAN.

    Notes
    -----
    This function assumes that the input DataFrame has a datetime column in the first position, and that the remaining
    columns are the values to be processed. The function will convert the first column to datetime if it's not already in
    that format.

    The function will add two new columns to the DataFrame: '..._IsoForest' and '..._DBSCAN', which will contain the
    outlier labels from Isolation Forest and DBSCAN, respectively.

    If `method` is set to 'both', the function will return a DataFrame with only the rows where both Isolation Forest and
    DBSCAN detect an outlier. If `method` is set to 'single', the function will return a DataFrame with only the rows where
    either Isolation Forest or DBSCAN detect an outlier.
    """
    # Get the first column (datetime) and remaining columns (values)
    datetime_column = df.columns[0]
    value_columns = df.columns[1:].tolist()

    # Convert first column to datetime
    df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')

    # Process each value column
    try:
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

            # TODO: Add code for specific method
            if method == 'both':
                mask = (df[col_name_iso] == -1) & (df[col_name_db] == -1)
            elif method == 'single':
                mask = (df[col_name_iso] == -1) | (df[col_name_db] == -1)
            print(f'mask: {mask}')
        return df[mask]
    except Exception as e:
        print(f'Error: {str(e)}')

# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# outliers = detect_outliers(df, 'single')
# print(f'Outliers: {outliers}')
# df.to_csv('ts_outliers.csv', index=False)
