import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


def normalize_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the given dataframe across all columns.
    :param dataframe: The dataframe to normalize.
    :return: The normalized dataframe.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_df = dataframe.copy()
    for column in normalized_df.columns[1:]:
        normalized_df[column] = normalized_df[column].replace(0, np.nan)
        normalized_df[column] = normalized_df[column].replace('', np.nan)
        normalized_df[column] = scaler.fit_transform(dataframe[[column]])
    return normalized_df


def cleanup_df_zero_nans(df: pd.DataFrame) -> pd.DataFrame:
    # Convert 0s and '' to nan
    cleaned_df = df.copy()
    for column in cleaned_df.columns[1:]:
        cleaned_df[column] = cleaned_df[column].replace(0, np.nan)
        cleaned_df[column] = cleaned_df[column].replace('', np.nan)

    # Drop rows with all nans or all 0s
    cleaned_df = cleaned_df.loc[:, ~((cleaned_df.isna() | (cleaned_df == 0)).all())]
    return cleaned_df


# TODO: Add different imputation per column
def impute_missing_data(df: pd.DataFrame, method: str):
    """
    Imputes missing values in a given dataframe using one of the following methods.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to impute missing values.
    method: str
        The method to use for imputation. Choices are:
        'fill', 'mean', 'median', 'most_frequent', 'moving_average' and 'linear_regression'.

    Returns
    -------
    pd.DataFrame
        The dataframe with missing values imputed.

    Notes
    -----
    For 'linear_regression', the time column must be in datetime format.
    """
    if method == 'fill':
        df = df.ffill().bfill()
    for column in df.columns[1:]:
        if method == 'mean':
            mean_imputer = SimpleImputer(strategy='mean')
            df[column] = mean_imputer.fit_transform(df[[column]])

        elif method == 'median':
            median_imputer = SimpleImputer(strategy='median')
            df[column] = median_imputer.fit_transform(df[[column]])

        elif method == 'most_frequent':
            most_frequent_imputer = SimpleImputer(strategy='most_frequent')
            df[column] = most_frequent_imputer.fit_transform(df[[column]])

        elif method == 'moving_average':
            df_copy = df.copy()
            df_copy[column] = df_copy[column].rolling(window=3, min_periods=1).mean()
            df[column] = df_copy[column].fillna(df_copy[column].mean())

        elif method == 'linear_regression':
            df_copy = df.copy()
            df_copy['time_numeric'] = (
                    pd.to_datetime(df_copy['time']) - pd.to_datetime(df_copy['time']).min()).dt.total_seconds()
            train_idx = df_copy[column].notna()

            X_train = df_copy.loc[train_idx, 'time_numeric'].values.reshape(-1, 1)
            y_train = df_copy.loc[train_idx, column].values

            model = LinearRegression()
            model.fit(X_train, y_train)

            X_all = df_copy['time_numeric'].values.reshape(-1, 1)
            predicted_values = model.predict(X_all)

            df.loc[df[column].isna(), column] = predicted_values[df[column].isna()]

    return df

# if __name__ == "__main__":
#     df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
#     df = cleanup_df_zero_nans(df)
#     print(df.head(20))
#     # Perform imputation on the raw DataFrame
#     imputed_df = impute_missing_data(df, 'moving_average')
#     print("\nAfter imputation (raw data):")
#     print(imputed_df.head(20))
#     normalized_df = normalize_data(imputed_df)
#     print("\nAfter normalization:")
#     print(normalized_df.head(20))
#
#     df.plot(x='time', y='value', kind='line')
#     plt.show()
