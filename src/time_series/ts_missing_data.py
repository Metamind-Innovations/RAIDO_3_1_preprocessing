import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def keep_data_avg(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe.set_index('time', inplace=True)
    daily_avg = dataframe.resample('D').mean()
    daily_avg.reset_index(inplace=True)
    daily_avg['time'] = daily_avg['time'].dt.strftime('%d/%m/%Y')
    daily_avg['value'] = daily_avg['value'].round(2)
    return daily_avg


def plot_imputed_data(dataframe: pd.DataFrame, method: str):
    data_no_zeros = dataframe['value'].replace(0, np.nan)
    if method == 'mean':
        mean_value = data_no_zeros.mean()
        data_imputed = dataframe.copy()
        data_imputed['value'] = data_imputed['value'].replace(0, mean_value)
    elif method == 'median':
        median_value = data_no_zeros.median()
        data_imputed = dataframe.copy()
        data_imputed['value'] = data_imputed['value'].replace(0, median_value)
    else:
        raise NotImplementedError(f"Invalid method: {method}")

    fig, ax = plt.subplots(3, figsize=(16, 8))
    fig.autofmt_xdate()
    plt.plot(dataframe['time'], dataframe['value'], label='Original Data', color='blue', alpha=0.5)

    # Plot imputed data in red for zero values, and same as original for non-zero values
    mask = dataframe['value'] == 0
    plt.plot(dataframe['time'][~mask], data_imputed['value'][~mask], color='blue', alpha=0.5)
    plt.plot(dataframe['time'][mask], data_imputed['value'][mask], color='red', linestyle='--')

    plt.title(f'Data Imputed with {method.capitalize()}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('power.csv', delimiter=';', parse_dates=['time'])
    df_avg = keep_data_avg(df)
    plot_imputed_data(df_avg, 'mean')
