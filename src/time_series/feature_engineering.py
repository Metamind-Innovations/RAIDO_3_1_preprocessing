from typing import List

import pandas as pd
import holidays
from sklearn.preprocessing import OneHotEncoder


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract date-related features from the given DataFrame.

    :param df: The DataFrame containing the date column as the first column.
    :type df: pd.DataFrame
    :return: The DataFrame with the new features added.
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()
    date_column = df_copy.columns[0]
    df_copy['year'] = df_copy[date_column].dt.year
    df_copy['month'] = df_copy[date_column].dt.month
    df_copy['day'] = df_copy[date_column].dt.day
    df_copy['dayofweek'] = df_copy[date_column].dt.dayofweek
    df_copy['dayofyear'] = df_copy[date_column].dt.dayofyear
    df_copy['quarter'] = df_copy[date_column].dt.quarter
    df_copy['is_weekend'] = (df_copy[date_column].dt.dayofweek >= 5).astype(int)
    df_copy['is_holiday'] = df_copy[date_column].apply(lambda x: x in holidays.US()).astype(int)
    return df_copy


def calculate_differences(df: pd.DataFrame, column: str = 'value', order: int = 1) -> pd.DataFrame:
    """
    Calculate the first and second order differences for the given column.

    :param df: The DataFrame containing the column.
    :type df: pd.DataFrame
    :param column: The name of the column.
    :type column: str
    :param order: The order of the differences. 1 for first order, 2 for second order.
    :type order: int
    :return: The DataFrame with the new features added.
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()

    # fillna used to handle nan values in order to be jsonable later
    for i in range(1, order + 1):
        df_copy[f'{column}_diff_{i}'] = df_copy[column].diff(i).fillna(0)
    return df_copy


def one_hot_encode_categoricals(df: pd.DataFrame, columns: List[str] = ['value']) -> pd.DataFrame:
    """
    One-hot encode the given columns.

    :param df: The DataFrame containing the columns.
    :type df: pd.DataFrame
    :param columns: The list of column names to encode.
    :type columns: List[str]
    :return: The DataFrame with the new features added.
    :rtype: pd.DataFrame
    """
    df_copy = df.copy()
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_df = pd.DataFrame(encoder.fit_transform(df_copy[columns]).toarray(),
                              columns=encoder.get_feature_names_out(columns))
    df_copy = pd.concat([df_copy, encoded_df], axis=1)
    df_copy = df_copy.drop(columns, axis=1)
    return df_copy


# df = pd.read_csv('power_small.csv', sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
# date_features = extract_date_features(df)
# print(f'Date Features: {date_features.head(20)}')
#
# differences = calculate_differences(df)
# print(f'Differences: {differences.head(20)}')
#
# one_hot = one_hot_encode_categoricals(df)
# print(f'one_hot: {one_hot.head(20)}')
