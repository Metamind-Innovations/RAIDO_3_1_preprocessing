import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


def normalize_data(dataframe: pd.DataFrame, column_name: str = 'value') -> pd.DataFrame:
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_df = dataframe.copy()
    normalized_df[column_name] = scaler.fit_transform(dataframe[[column_name]])
    return normalized_df

def impute_missing_data(df: pd.DataFrame, method: str, column_name='value'):
    if method == 'mean':
        mean_imputer = SimpleImputer(strategy='mean')
        df[column_name] = mean_imputer.fit_transform(df[[column_name]])
        return df

    elif method == 'median':
        median_imputer = SimpleImputer(strategy='median')
        df[column_name] = median_imputer.fit_transform(df[[column_name]])
        return df

    elif method == 'most_frequent':
        most_frequent_imputer = SimpleImputer(strategy='most_frequent')
        df[column_name] = most_frequent_imputer.fit_transform(df[[column_name]])
        return df

    elif method == 'linear_regression':
        df_copy = df.copy()
        df_copy['time_numeric'] = (
                    pd.to_datetime(df_copy['time']) - pd.to_datetime(df_copy['time']).min()).dt.total_seconds()
        train_idx = df_copy[column_name].notna()

        X_train = df_copy.loc[train_idx, 'time_numeric'].values.reshape(-1, 1)
        y_train = df_copy.loc[train_idx, column_name].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        X_all = df_copy['time_numeric'].values.reshape(-1, 1)
        predicted_values = model.predict(X_all)

        df.loc[df[column_name].isna(), column_name] = predicted_values[df[column_name].isna()]
        return df

    return df


if __name__ == "__main__":
    df = pd.read_csv('power.csv', sep=';', parse_dates=['time'], dayfirst=True)
    # Replace '0' with NaN (missing value)
    df['value'] = df['value'].replace(0, np.nan)
    # Normalize 'values' column
    df = normalize_data(df)
    df = impute_missing_data(df, 'mean', 'value')
    df.plot(x='time', y='value', figsize=(15, 6))
    plt.show()
